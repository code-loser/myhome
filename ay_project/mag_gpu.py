import argparse
import torch.multiprocessing as mp
import multiprocessing
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
import time
import tqdm
import argparse
import pandas as pd
import dgl
import numpy as np
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler
import dgl.nn.pytorch as dglnn
import storage_cpu3
import sys
sys.path.append('/ay_project/cuda_ipc')
import cuda_ipc_open
import cuda_ipc_get
import cuda_ipc_close
import base64
import cupy as cp
import os
from ogb.lsc import MAG240MDataset
from queue import Queue
import psutil
from models.gat import GAT
from models.sagenet import SAGENet
from models.gcn import GCN


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
            blocks = [b.to(device) for b in blocks]
        else:
            x = x_cpu
            y = y_cpu
            blocks = [b.to(device) for b in blocks]

        if use_bf16:
            x = x.to(torch.bfloat16)

        ys.append(y)
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


def train(args, device, g, model_list):
        

    def get_iter(visibility):
        for i in range(args.vis):
            input_nodes, output_nodes, blocks = next(dataloader_iter)
            result_nodes[i*args.avg_input_size:i*args.avg_input_size + input_nodes.numel()].copy_(input_nodes.long())
            offset_list[i] = input_nodes.numel()
            blocks_queue.put(blocks)
            output_nodes_queue.put(output_nodes)
    tensor_label_all = torch.full((g.number_of_nodes(), ), fill_value=1,dtype=torch.long, device=device)
    
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    
    sampler = NeighborSampler(
        [15 ,15, 15],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        #prefetch_labels=["label"],
    )
    hotmap_loader = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # use_uva=use_uva,
    )
    sub_dataloader = DataLoader(
            g,
            train_idx,
            sampler,
            device="cpu",
            batch_size=train_idx.numel(),
            shuffle=True,
            drop_last=False,
            num_workers=0,
            # use_uva=use_uva,
    )
    mp.set_start_method(method='forkserver', force=True)
    handle_dict = multiprocessing.Queue()
    task_queue = multiprocessing.Queue()
    signal = multiprocessing.Queue()
    p = mp.Process(target=storage_cpu3.cache_worker_main, args=(args , g.num_nodes(), 1 , handle_dict, task_queue, signal))
    p.start()
    p = psutil.Process()
    cores_socket_0 = list(range(0,20))
    try:
        p.cpu_affinity(cores_socket_0)
    except Exception as e:
        print(f"❌ Failed to set CPU affinity: {e}")
    torch.set_num_threads(20)
    num_threads = "20"
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    start_time = time.time()
    graph_nodes , graph_node , graph_blocks = next(iter(sub_dataloader))
    covered_nodes = train_idx.clone()
    seed = train_idx.clone()
    train_neighbors = g.in_edges(seed)[0].unique()
    covered_nodes = torch.cat([covered_nodes, train_neighbors]).unique()
    sub_g = dgl.node_subgraph(g, graph_nodes,relabel_nodes=False)
    degrees = torch.full((2*g.num_nodes(),), fill_value=-1, dtype=torch.int64)
    degrees[:g.num_nodes()][covered_nodes] = sub_g.out_degrees()[covered_nodes]
    dataloader_iter_pre = iter(hotmap_loader)
    for i in range(len(hotmap_loader)):
        input_nodes, output_nodes, blocks = next(dataloader_iter_pre)
        degrees[input_nodes] += 1
    torch.cuda.set_device(1)
    degrees.share_memory_()
    task_queue.put(degrees)
    key_dict = ['train_mask', 'val_mask', 'test_mask' , 'label']
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device="cpu",
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    while handle_dict.empty():
        pass
    item = handle_dict.get()
    handle_1 ,  shape1 = item
    blocks_queue = Queue()
    output_nodes_queue = Queue()
    handle_feature = base64.b64decode(handle_1)  # bytes
    tensor_feature_all = cuda_ipc_open.open_cuda_ipc_handle(handle_feature, list(shape1), 1 , "float32")
    vis = args.vis + 1
    result_nodes = torch.empty(vis * args.avg_input_size, dtype=torch.long)
    offset_list = torch.empty((vis+1),dtype=torch.long)
    result_nodes.share_memory_()
    offset_list.share_memory_()
    task_queue.put((result_nodes , offset_list), block=False)
    print("start training")
    use_bf16 = (args.dt == "bfloat16")
    model_name = [ "SAGE","GAT","GCN"]
    for model_index in range(0,3):
        model = model_list[model_index]
        opt = torch.optim.Adam(
            model.parameters(), 
            lr=0.003,            
            weight_decay=0.0005  
        )
        torch.cuda.synchronize()
        total_epochs = args.epochs + 1
        all_epoch_sampling_ms = []
        all_epoch_copy_ms = []
        all_epoch_forward_ms = []
        all_epoch_backward_ms = []
        all_epoch_total_ms = []
        save_dir = os.path.join(model_name[model_index])
        os.makedirs(save_dir, exist_ok=True)
        iter_log_path = os.path.join(save_dir, "mag-log.txt")
        if args.log_time:
            with open(iter_log_path, "a", encoding="utf-8") as f:
                f.write("")
        torch.cuda.set_device(device)
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
            get_iter(vis)
            task_queue.put(len(train_dataloader),block = True)
            while handle_dict.empty():
                  pass
            for it in range(num_iters):
                # ---------------- 1) SamplingOnly ----------------
                if args.log_time:
                    t0 = time.time()
                try:
                    next_input_nodes, next_output_nodes, next_blocks = next(dataloader_iter)
                    idx = (it-1) % vis
                    result_nodes[idx*args.avg_input_size:idx*args.avg_input_size + next_input_nodes.numel()] = next_input_nodes
                    signal.put(0 ,block=False)
                    offset_list[idx] = next_input_nodes.numel()
                    output_nodes_queue.put(next_output_nodes)
                    blocks_queue.put(next_blocks)
                except StopIteration:
                    pass
                if args.log_time:
                    sampling_ms = (time.time() - t0) * 1000.0
                else:
                    sampling_ms = 0.0
                blocks = blocks_queue.get()
                output_nodes = output_nodes_queue.get()

                # ---------------- 3) BlockH2D ----------------
                if args.log_time and device.type == "cuda":
                    blocks, block_h2d_ms = time_cuda_op(
                        lambda: [b.to(device) for b in blocks], device
                    )
                else:
                    if args.log_time:
                        t0 = time.time()
                    blocks = [b.to(device) for b in blocks]
                    block_h2d_ms = (time.time() - t0) * 1000.0 if args.log_time else 0.0

                
                # ---------------- 2) FeatH2D ----------------
                if args.log_time and device.type == "cuda":
                    def _move_feat_label():
                        while handle_dict.empty():
                            pass
                        item = handle_dict.get()
                        x = tensor_feature_all[0:offset_list[(it%vis)]].float()
                        y = g.ndata["label"][output_nodes].to(device)
                        y = y.long()
                        return x, y

                    (x, y), feat_h2d_ms = time_cuda_op(_move_feat_label, device)
                else:
                    if args.log_time:
                        t0 = time.time()
                    x_cpu = g.ndata["feat"][input_nodes]
                    y_cpu = g.ndata["label"][output_nodes].view(-1)
                    x = x_cpu.to(device) if device.type == "cuda" else x_cpu
                    y = y_cpu.long().to(device) if device.type == "cuda" else y_cpu.long()
                    feat_h2d_ms = (time.time() - t0) * 1000.0 if args.log_time else 0.0

                copy_ms = feat_h2d_ms + block_h2d_ms

                # ---------------- 4) Forward ----------------
                if args.log_time and device.type == "cuda":
                    y_hat, forward_time_ms = time_cuda_op(lambda: model(blocks, x), device)
                else:
                    if args.log_time:
                        t0 = time.time()
                    y_hat = model(blocks, x)
                    forward_time_ms = (time.time() - t0) * 1000.0 if args.log_time else 0.0

                loss = F.cross_entropy(y_hat, y)

                # ---------------- 5) Backward ----------------
                if args.log_time and device.type == "cuda":
                    def _backward_step():
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    _, backward_time_ms = time_cuda_op(_backward_step, device)
                else:
                    if args.log_time:
                        t0 = time.time()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    backward_time_ms = (time.time() - t0) * 1000.0 if args.log_time else 0.0

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

            all_epoch_sampling_ms.append(epoch_sampling_ms)
            all_epoch_copy_ms.append(epoch_copy_ms)
            all_epoch_forward_ms.append(epoch_forward_time_ms)
            all_epoch_backward_ms.append(epoch_backward_time_ms)
            all_epoch_total_ms.append(epoch_total_time_ms)

            # val_acc = evaluate(model, g, val_dataloader, num_classes, device, use_bf16)

            epoch_tag = "warm-up" if epoch == 0 else "measured"
            # print(
            #     f"Epoch {epoch:03d} ({epoch_tag}) | "
            #     f"Loss {total_loss / max(1, num_iters):.6f} | "
            #     f"ValAcc {val_acc:.6f}"
            # )

            if args.log_time:
                print(f"  Avg SamplingOnly: {epoch_sampling_ms / max(1, num_iters):.4f} ms/iter")
                print(f"  Avg Copy       : {epoch_copy_ms / max(1, num_iters):.4f} ms/iter")
                print(f"  Avg Forward    : {epoch_forward_time_ms / max(1, num_iters):.4f} ms/iter")
                print(f"  Avg Backward   : {epoch_backward_time_ms / max(1, num_iters):.4f} ms/iter")
                print(f"  Avg Total      : {epoch_total_time_ms / max(1, num_iters):.4f} ms/iter")
                print("########## EPOCH SUMMARY END ##########")
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
    parser.add_argument('--feat_name', type=str, default="mag",
        help='feature name')
    parser.add_argument('--avg_input_size', type=int, default=350000,
        help='average input size')
    parser.add_argument('--use_uva', type=bool, default=False,
        help='use uva or not')
    parser.add_argument('--dtype', type=str, default="float32",
        help='use uva or not')

    args = parser.parse_args()
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
    
    paper_label = torch.from_numpy(dataset.paper_label).long()
    
    
    g.ndata['label'] = paper_label
    
    
    print("Creating train/val/test splits...")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    
    # 将验证集分成两份：一半作为验证集，一半作为测试集
    valid_size = len(valid_idx)
    val_idx = valid_idx[:valid_size//2]
    test_idx = valid_idx[valid_size//2:]
    
    # 创建掩码
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
    model_gat = GAT(in_size, 256, out_size, num_heads=4).to(device)
    model_gcn = GCN(in_size, 256, out_size).to(device)
    model_list = [model_sage,model_gat,model_gcn]
    print("Training...")
    train(args, device, g, model_list)