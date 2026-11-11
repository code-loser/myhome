import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import time
import tqdm
import argparse
import pandas as pd
import dgl
import numpy as np
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler
import dgl.nn.pytorch as dglnn
from storage import BlockFeatureCacheServer
from dgl.data import RedditDataset
import dgl.graphbolt as gb
import os.path as osp
from torch_geometric.datasets import Yelp
import torch_geometric.transforms as T
from ogb.lsc import MAG240MDataset
import os
from torch.cuda.amp import autocast, GradScaler

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()

        self.layers = nn.ModuleList()

        
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu , allow_zero_in_degree=True)
        )

        
        self.layers.append(
            dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )

        
        self.layers.append(
            dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True)
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            
            h = layer(block, h)

            if l != len(self.layers) - 1:
                h = self.dropout(h)

        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads=4):
        super().__init__()

        self.layers = nn.ModuleList()

       
        self.layers.append(
            dglnn.GATConv(
                in_size,
                hid_size // num_heads,   
                num_heads,               
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        
        self.layers.append(
            dglnn.GATConv(
                hid_size,                 
                hid_size // num_heads,
                num_heads,
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        
        self.layers.append(
            dglnn.GATConv(
                hid_size,
                out_size,
                num_heads=1,
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # h shape = (N, num_heads, out_dim)
            h = h.flatten(1)

            if l != len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)

        return h
    
@torch.no_grad()
def evaluate(model, graph, dataloader, num_classes, device, use_bf16: bool):
    model.eval()
    ys, y_hats = [], []

    for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x_cpu = graph.ndata["feat"][input_nodes]
        y_cpu = graph.ndata["label"][output_nodes].view(-1)

        if device.type == "cuda":
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            x = x.float()
            blocks = [b.to(device) for b in blocks]
        else:
            x = x_cpu
            y = y_cpu
            blocks = [b.to(device) for b in blocks]

        if use_bf16:
            x = x.to(torch.bfloat16)

        ys.append(y)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            y_hats.append(model(blocks, x))

    y_true = torch.cat(ys)
    y_pred = torch.cat(y_hats).argmax(dim=1)

    acc = (y_true == y_pred).float().mean().item()
    return acc

def time_cuda_op(fn, device):
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    result = fn()
    end_evt.record()
    torch.cuda.synchronize(device)
    return result, start_evt.elapsed_time(end_evt)

def time_cuda_op1(fn, device):
    result = fn()
    return result
def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_index(idx_full: torch.Tensor, frac: float) -> torch.Tensor:
    n = idx_full.shape[0]
    k = max(1, int(n * frac))
    perm = torch.randperm(n)[:k]
    return idx_full[perm]


def mean_skip_warmup(xs):
    if len(xs) <= 1:
        return sum(xs) / len(xs) if len(xs) > 0 else 0.0
    ys = xs[1:]
    return sum(ys) / len(ys) if len(ys) > 0 else 0.0


def sum_skip_warmup(xs):
    if len(xs) <= 1:
        return sum(xs)
    return sum(xs[1:])


def train(args, device, g, model_list, num_classes ,sampler_type):
    def fetch_from_cache(gpuid, blocks, input_nodes ,output_nodes):
        if input_nodes.device.type == "cuda":
            input_nodes = input_nodes.cpu()
            output_nodes = output_nodes.cpu()
        on_cache_mask = cache_bucket[input_nodes] != -1
        nids_cpu = input_nodes[~on_cache_mask]
        nids_cuda = input_nodes[on_cache_mask]
        feature_fetch = torch.empty((input_nodes.numel(), g.ndata['feat'].shape[1]), dtype=torch.float32, device=device)
        feature_fetch[on_cache_mask] = gpu_fix_cache[cache_bucket[nids_cuda].to(device)]
        feature_fetch[~on_cache_mask] = g.ndata['feat'][nids_cpu].to(device)
        blocks[0].srcdata["feat"] = feature_fetch
        blocks[-1].dstdata["label"] = g.ndata['label'][output_nodes].to(device)

    def prefill_by_degree():
        degrees = sub_g.out_degrees()
        cache_size = int(g.num_nodes() * 0.005)
        print(f"Prefill {cache_size} nodes")
        topk_vals, topk_idx = torch.topk(degrees, k=cache_size, largest=True)
        topk_idx = topk_idx.long()
        slots = torch.arange(0, cache_size, dtype=torch.long)
        feats = g.ndata['feat'][topk_idx].to(device, non_blocking=True)
        gpu_fix_cache[slots.to(device)] = feats
        cache_bucket[topk_idx] = slots

    def prefill_by_random():
        cache_size = int(g.num_nodes() * 0.005)
        topk_idx = torch.randint(0, g.num_nodes(), size=(cache_size,), dtype=torch.long)
        topk_idx = topk_idx.long()
        slots = torch.arange(0, cache_size, dtype=torch.long)
        feats = g.ndata['feat'][topk_idx].to(device, non_blocking=True)
        gpu_fix_cache[slots.to(device)] = feats
        cache_bucket[topk_idx] = slots

    def prefill_by_hotmap(train_dataloader):
        dataloader_iter1 = iter(train_dataloader)
        hot_map = torch.zeros(g.num_nodes())
        for i in range(len(train_dataloader)):
            input_nodes, output_nodes, blocks = next(dataloader_iter1)
            hot_map[input_nodes] += 1
        cache_size = int(g.num_nodes() * cache_ratio)
        print(f"Prefill {cache_size} nodes")
        topk_idx = torch.topk(hot_map, k=cache_size, largest=True)[1]
        topk_idx = topk_idx.long()
        slots = torch.arange(0, cache_size, dtype=torch.long)
        feats = g.ndata['feat'][topk_idx].to(device, non_blocking=True)
        cache_bucket[topk_idx] = slots
        return feats

    gpu_fix_cache = torch.empty((int(g.num_nodes() * 0.002) , g.ndata['feat'].shape[1]), dtype=torch.float32, device=device)
    cache_bucket = torch.full(
            (g.num_nodes(),),
            fill_value=-1,
            dtype=torch.long,
        )
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_idx = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    sampler = NeighborSampler(
        [15 ,15, 15],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        #prefetch_labels=["label"],
    )
    sub_sampler = NeighborSampler(
        [-1 ,-1, -1],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        #prefetch_labels=["label"],
    )
    sub_loader = DataLoader(
        g,
        train_idx,
        sub_sampler,
        device="cpu",
        batch_size=train_idx.shape[0],
        shuffle=False,
        drop_last=False,
        num_workers=0,
        # use_uva=use_uva,
    )
    train_dataloader1 = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    torch.cuda.set_device(1)
    # prefill_by_hotmap(train_dataloader)
    
    for graph_nodes, output_nodes, blocks in sub_loader:
        sub_g = dgl.node_subgraph(g, graph_nodes,relabel_nodes=False)
    key_dict = ['train_mask', 'val_mask', 'test_mask' , 'label','feat']
    for key in key_dict:
        del sub_g.ndata[key]
    init_time_start = time.time()
    sub_g.create_formats_()
    sub_g = sub_g.formats(['csc'])
    sub_g = sub_g.int()
    sub_g = sub_g.to(device)
    torch.cuda.synchronize()
    init_time = time.time() - init_time_start
    save_dir = "sage_log"
    os.makedirs(save_dir, exist_ok=True)
    iter_log_path = os.path.join(save_dir, "mag-log.txt")
    
    train_dataloader = DataLoader(
        sub_g,
        train_idx.int().to(device),
        sampler,
        device=device,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # val_dataloader = DataLoader(
    #     g,
    #     val_idx,
    #     sampler,
    #     device="cpu",
    #     batch_size=2048,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    # )
    
    use_bf16 = (args.dt == "bfloat16")

    total_epochs = args.epochs + 1

    # epoch-level records
    all_epoch_sampling_ms = []
    all_epoch_copy_ms = []
    all_epoch_forward_ms = []
    all_epoch_backward_ms = []
    all_epoch_total_ms = []

    if args.log_time:
        with open(iter_log_path, "a", encoding="utf-8") as f:
            f.write("")
    prefill_list = [prefill_by_hotmap]
    for prefill_index, prefill in enumerate(prefill_list):
        if prefill_index == 2:
            cache_ratio = 0
        else:
            cache_ratio = 0.002
        init_time1_start = time.time()
        cache_bucket = torch.full(
                (g.num_nodes(),),
                fill_value=-1,
                dtype=torch.long,
            )
        gpu_fix_cache = prefill(train_dataloader1)
        init_time1 = time.time() - init_time1_start
        print(f"prefill time: {init_time1 + init_time}")
           
        for model in model_list:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            for epoch in range(total_epochs):
                model.train()
                total_loss = 0.0
                num_iters = len(train_dataloader)

                epoch_sampling_ms = 0.0
                epoch_copy_ms = 0.0
                epoch_forward_time_ms = 0.0
                epoch_backward_time_ms = 0.0
                epoch_total_time_ms = 0.0

                it_now = 0
                dataloader_iter = iter(train_dataloader)
                start_time = time.time() 
                for _ in range(num_iters):
                    # ---------------- 1) SamplingOnly ----------------
                    if args.log_time:
                        t0 = time.time()
                    input_nodes, output_nodes, blocks = next(dataloader_iter)
                    if args.log_time:
                        sampling_ms = (time.time() - t0) * 1000.0
                    else:
                        sampling_ms = 0.0

                    num_input_nodes = int(input_nodes.numel())
                    num_output_nodes = int(output_nodes.numel())
                    approx_feat_mb = num_input_nodes*g.ndata["feat"].shape[1]*g.ndata["feat"].element_size() / (1024 * 1024)

                    # ---------------- 3) BlockH2D ----------------
                    if args.log_time and device.type == "cuda":
                        blocks, block_h2d_ms = time_cuda_op(
                            lambda: [b.to(device) for b in blocks], device
                        )
                    else:
                        blocks = [b.to(device) for b in blocks]

                    
                    # ---------------- 2) FeatH2D ----------------
                    def _move_feat_label():
                        fetch_from_cache(1,blocks, input_nodes, output_nodes)
                        x = blocks[0].srcdata["feat"]
                        y = blocks[-1].dstdata["label"]
                        return x, y
                    if args.log_time and device.type == "cuda":
                        (x, y), feat_h2d_ms = time_cuda_op(_move_feat_label, device)
                        copy_ms = feat_h2d_ms
                    else:
                        (x,y) = time_cuda_op1(_move_feat_label, device)


                    # ---------------- 4) Forward ----------------
                    if args.log_time and device.type == "cuda":
                        y_hat, forward_time_ms = time_cuda_op(lambda: model(blocks, x), device)
                        # y_hat = time_cuda_op1(lambda: model(blocks, x), device)
                    else:
                        y_hat = time_cuda_op1(lambda: model(blocks, x), device)

                    loss = F.cross_entropy(y_hat, y)

                    # ---------------- 5) Backward ----------------
                    def _backward_step():
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    if args.log_time :
                        _, backward_time_ms = time_cuda_op(_backward_step, device)
                    else:
                        _ = time_cuda_op1(_backward_step, device)
                    
                    if args.log_time:
                        total_time_ms = sampling_ms + copy_ms + forward_time_ms + backward_time_ms

                        # accumulate
                        epoch_sampling_ms += sampling_ms
                        epoch_copy_ms += copy_ms
                        epoch_forward_time_ms += forward_time_ms
                        epoch_backward_time_ms += backward_time_ms
                        epoch_total_time_ms += total_time_ms

                    # if args.log_time:
                    #     with open(iter_log_path, "a", encoding="utf-8") as f:
                    #         f.write("========================================\n")
                    #         f.write(f"Epoch: {epoch}\n")
                    #         f.write(f"Iter: {it_now}\n")
                    #         f.write(f"Input Nodes: {num_input_nodes}\n")
                    #         f.write(f"Output Nodes: {num_output_nodes}\n")
                    #         f.write(f"Approx Feature Size: {approx_feat_mb:.2f} MB\n")
                    #         f.write(f"SamplingOnly Time: {sampling_ms:.4f} ms\n")
                    #         f.write(f"Copy Time: {copy_ms:.4f} ms\n")
                    #         f.write(f"Forward Time: {forward_time_ms:.4f} ms\n")
                    #         f.write(f"Backward Time: {backward_time_ms:.4f} ms\n")
                    #         f.write(f"Total Time: {total_time_ms:.4f} ms\n")
                    #         f.write(f"Loss: {loss.item():.6f}\n")

                    it_now += 1
                    total_loss += loss.item()
                if args.log_time:
                    all_epoch_sampling_ms.append(epoch_sampling_ms)
                    all_epoch_copy_ms.append(epoch_copy_ms)
                    all_epoch_forward_ms.append(epoch_forward_time_ms)
                    all_epoch_backward_ms.append(epoch_backward_time_ms)
                    all_epoch_total_ms.append(epoch_total_time_ms)
                
                else:
                    torch.cuda.synchronize()
                    end = time.time()
                    print(f"Epoch time{end - start_time} | ")

                # val_acc = evaluate(model, g, val_dataloader, num_classes, device, use_bf16)

                epoch_tag = "warm-up" if epoch == 0 else "measured"
                print(
                    f"Epoch {epoch:03d} ({epoch_tag}) | "
                    f"Loss {total_loss / max(1, num_iters):.6f} | "
                    # f"ValAcc {val_acc:.6f}"
                )

                if args.log_time:
                    print(f"  Avg SamplingOnly: {epoch_sampling_ms / max(1, num_iters):.4f} ms/iter")
                    print(f"  Avg Copy       : {epoch_copy_ms / max(1, num_iters):.4f} ms/iter")
                    print(f"  Avg Forward    : {epoch_forward_time_ms / max(1, num_iters):.4f} ms/iter")
                    print(f"  Avg Backward   : {epoch_backward_time_ms / max(1, num_iters):.4f} ms/iter")
                    print(f"  Avg Total      : {epoch_total_time_ms / max(1, num_iters):.4f} ms/iter")

                    with open(iter_log_path, "a", encoding="utf-8") as f:
                        f.write("########## EPOCH SUMMARY ##########\n")
                        f.write(f"Epoch: {epoch} ({epoch_tag})\n")
                        f.write(f"epoch_sampling_ms: {epoch_sampling_ms:.4f}\n")
                        f.write(f"epoch_copy_ms: {epoch_copy_ms:.4f}\n")
                        f.write(f"epoch_forward_time_ms: {epoch_forward_time_ms:.4f}\n")
                        f.write(f"epoch_backward_time_ms: {epoch_backward_time_ms:.4f}\n")
                        f.write(f"epoch_total_time_ms: {epoch_total_time_ms:.4f}\n")
                        f.write(f"epoch_avg_sampling_ms: {epoch_sampling_ms / max(1, num_iters):.4f}\n")
                        f.write(f"epoch_avg_copy_ms: {epoch_copy_ms / max(1, num_iters):.4f}\n")
                        f.write(f"epoch_avg_forward_ms: {epoch_forward_time_ms / max(1, num_iters):.4f}\n")
                        f.write(f"epoch_avg_backward_ms: {epoch_backward_time_ms / max(1, num_iters):.4f}\n")
                        f.write(f"epoch_avg_total_ms: {epoch_total_time_ms / max(1, num_iters):.4f}\n")
                        # f.write(f"val_acc: {val_acc:.6f}\n")
                        f.write("###################################\n")

            if args.log_time:
                print("\nMeasured Epoch Averages (skip warm-up):")
                print(f"  Avg SamplingOnly : {mean_skip_warmup(all_epoch_sampling_ms):.2f} ms")
                print(f"  Avg Copy         : {mean_skip_warmup(all_epoch_copy_ms):.2f} ms")
                print(f"  Avg Forward      : {mean_skip_warmup(all_epoch_forward_ms):.2f} ms")
                print(f"  Avg Backward     : {mean_skip_warmup(all_epoch_backward_ms):.2f} ms")
                print(f"  Avg Total        : {mean_skip_warmup(all_epoch_total_ms):.2f} ms")

                print("\nTotal Timings Across Measured Epochs (skip warm-up):")
                print(f"  SamplingOnly : {sum_skip_warmup(all_epoch_sampling_ms):.2f} ms")
                print(f"  Copy         : {sum_skip_warmup(all_epoch_copy_ms):.2f} ms")
                print(f"  Forward      : {sum_skip_warmup(all_epoch_forward_ms):.2f} ms")
                print(f"  Backward     : {sum_skip_warmup(all_epoch_backward_ms):.2f} ms")
                print(f"  Total        : {sum_skip_warmup(all_epoch_total_ms):.2f} ms")

class CustomDataset:
    def __init__(self, g, train_idx, val_idx, test_idx, num_classes):
        self.graph = g
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.num_classes = num_classes
    def __getitem__(self, i):
        return self.graph

class SAGENet(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGENet, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = dglnn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = dglnn.SAGEConv(h_feats, num_classes, 'mean')
        self.dropout = torch.nn.Dropout(0.5)
        self.h_feats = h_feats

    def forward(self, blocks, x):
        h = x
        h = self.conv1(blocks[0], h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(blocks[1], h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv3(blocks[2], h)
        return h

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    
    # Loading dataset
    parser.add_argument('--path', type=str, default='/data/IGB-dataset/', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='large',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--log_time', type=bool, default=True,help='log time')
    parser.add_argument('--sample', type=str, default="nei",
        choices=["random_walk", "nei" , "weighted"])
    parser.add_argument('--sample_ratio', type=float, default=0.6,
        help='sampling ratio for train/val/test sets (default: 0.01)')
    parser.add_argument('--vis', type=int, default=6,
        help='cache vis')
    parser.add_argument('--feat_name', type=str, default="IGB260M",
        help='feature name')

    args = parser.parse_args()
    
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    print("Loading MAG240M data...")
    dataset = MAG240MDataset(root="/data")

    edge_index_cites = dataset.edge_index('paper', 'paper')
    
    print("Creating DGL graph...")
    src = torch.from_numpy(edge_index_cites[0])
    dst = torch.from_numpy(edge_index_cites[1])
    g = dgl.graph((src, dst), num_nodes=dataset.num_papers)
    print("Setting node features and labels...")
    paper_feat = torch.from_numpy(dataset.paper_feat)
    paper_label = torch.from_numpy(dataset.paper_label).long()
    
    g.ndata['feat'] = paper_feat
    g.ndata['label'] = paper_label

    print("Creating train/val/test splits...")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    
    valid_size = len(valid_idx)
    val_idx = valid_idx[:valid_size//2]
    test_idx = valid_idx[valid_size//2:]
    
    train_mask = torch.zeros(dataset.num_papers, dtype=torch.bool)
    val_mask = torch.zeros(dataset.num_papers, dtype=torch.bool)
    test_mask = torch.zeros(dataset.num_papers, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    device = torch.device("cpu" if args.mode == "cpu" else "cuda:1")
    
    num_classes = 153

    # create GraphSAGE model
    in_size = 768
    out_size = num_classes
    model_sage = SAGE(in_size, 256, out_size).to(device)
    model_gcn = GCN(in_size, 256, out_size).to(device)
    model_gate = GAT(in_size, 256, out_size, num_heads=4).to(device)
    model_list = [model_sage]
    # model training
    print("Training...")
    train(args, device, g, model_list, num_classes ,args.sample)