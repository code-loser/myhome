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
from ogb.nodeproppred import DglNodePropPredDataset
import time
import tqdm
import argparse
import pandas as pd
import dgl
import numpy as np
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler
import dgl.nn.pytorch as dglnn
import storage_cpu_gate
import sys
sys.path.append('/ay_project/cuda_ipc')
import cuda_ipc_open
import cuda_ipc_get
import cuda_ipc_close
import base64
import cupy as cp
import os
from igb_dataloader import IGB260MDGLDataset_part
from queue import Queue
from dgl.sampling import random_walk
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

def train(args, device, g, model, num_classes ,sampler_type):

        

    def get_iter(visibility):
        for i in range(args.vis):
            input_nodes, output_nodes, blocks = next(dataloader_iter)
            mask = (cold_node_mapp[input_nodes] == 0)
            gate_mask[i*args.avg_input_size:i*args.avg_input_size + input_nodes.numel()] = mask
            real_len[i] = input_nodes.numel()
            input_nodes_queue.put(input_nodes[~mask].cpu())
            input_nodes = input_nodes[mask]
            result_nodes[i*args.avg_input_size:i*args.avg_input_size + input_nodes.numel()].copy_(input_nodes.long())
            offset_list[i] = input_nodes.numel()
            blocks_queue.put(blocks)
            output_nodes_queue.put(output_nodes)
            
    def test_iter(hot_map_loader):
        dataloader_iter_pre = iter(hot_map_loader)
        for i in range(100):
            input_nodes, output_nodes, blocks = next(dataloader_iter_pre)
            mask = (cold_node_mapp[input_nodes] == 0)
            print(input_nodes.numel() , mask.sum().item())
    
    train_idx_full = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0].to("cpu")
    val_idx_full = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0].to("cpu")
    test_idx_full = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0].to("cpu")
    seed_everything(1)
    frac = float(0.01)
    train_idx = sample_index(train_idx_full, frac=frac)
    val_idx = sample_index(val_idx_full, frac=frac)
    test_idx = sample_index(test_idx_full, frac=frac)
    torch.cuda.set_device(device)
    print(f"Sampled sizes (frac={frac}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    sampler = NeighborSampler(
        [5 ,10, 15],  # fanout for [layer-0, layer-1, layer-2]
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
    degrees = torch.zeros(g.num_nodes(), dtype=torch.long)
    dataloader_iter_pre = iter(hotmap_loader)
    for graph_nodes, output_nodes, blocks in sub_loader:
        sub_g = dgl.node_subgraph(g, graph_nodes,relabel_nodes=False)
    key_dict = ['train_mask', 'val_mask', 'test_mask' , 'label']
    for key in key_dict:
        del sub_g.ndata[key]
    sub_g.create_formats_()
    sub_g = sub_g.formats(['csc'])
    sub_g = sub_g.int() 
    sub_g = sub_g.to(device)
    for i in range(len(hotmap_loader)):
        input_nodes, output_nodes, blocks = next(dataloader_iter_pre)
        degrees[input_nodes] += 1
    init_time_start = time.time()
    num_node_in = torch.where(degrees[:g.num_nodes()] > 0)[0].numel()
    values, indices = torch.topk(degrees, int(0.48*num_node_in))
    hot_gate = values[-1]
    torch.cuda.set_device(1)
    degrees.share_memory_()
    cold_node_mapp = torch.full((g.num_nodes(), ), fill_value = 0,dtype=torch.bool ,device = device)
    code_node = torch.where((degrees[:g.num_nodes()] <= hot_gate) & (degrees[:g.num_nodes()] > 0))[0]
    cold_node_mapp[code_node] = 1
    print(num_node_in ,code_node.numel())
    
    dataset = IGB260MDGLDataset_part(args , only_feat=True)
    ndata = dataset[0]
    
    # args.limits = len(train_dataloader)
    mp.set_start_method(method='forkserver', force=True)
    handle_dict = multiprocessing.Queue()
    task_queue = multiprocessing.Queue()
    signal = multiprocessing.Queue()
    p = mp.Process(target=storage_cpu_gate.cache_worker_main, args=(args , g.num_nodes(), 1 , handle_dict, task_queue, signal))
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
    pinned_buffer = torch.empty((args.avg_input_size , 1024), dtype=torch.float32).pin_memory()
    start_time = time.time()
    task_queue.put(degrees)
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
    input_nodes_queue = Queue()
    handle_feature = base64.b64decode(handle_1)  # bytes
    tensor_feature_all = cuda_ipc_open.open_cuda_ipc_handle(handle_feature, list(shape1), 1 , args.dtype)
    vis = args.vis + 1
    result_nodes = torch.empty(vis * args.avg_input_size, dtype=torch.long)
    real_len = torch.empty(vis , dtype=torch.long)
    offset_list = torch.empty(vis,dtype=torch.long)
    gate_mask = torch.zeros(vis*args.avg_input_size, dtype=torch.bool)
    gate_mask.share_memory_()
    result_nodes.share_memory_()
    offset_list.share_memory_()
    task_queue.put((result_nodes , offset_list , gate_mask, real_len), block=False)
    init_time = time.time() - init_time_start
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
        iter_log_path = os.path.join(save_dir, "igb-log.txt")
        if args.log_time:
            with open(iter_log_path, "a", encoding="utf-8") as f:
                f.write("")
        
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
            init_time1 = time.time()
            get_iter(vis)
            task_queue.put(len(train_dataloader),block = True)
            while handle_dict.empty():
                  pass
            start_time = time.time()
            init_end = start_time - init_time1
            print(f"Initialization Time:{init_end + init_time}")
            for it in range(num_iters):
                # ---------------- 1) SamplingOnly ----------------
                t0 = time.time()
                try:
                    next_input_nodes, next_output_nodes, next_blocks = next(dataloader_iter)
                    mask = (cold_node_mapp[next_input_nodes] == 0)
                    idx = (it-1) % vis
                    gate_mask[idx*args.avg_input_size:idx*args.avg_input_size + next_input_nodes.numel()] = mask
                    real_len[idx] = next_input_nodes.numel()
                    input_nodes_queue.put(next_input_nodes[~mask].cpu())
                    next_input_nodes = next_input_nodes[mask]
                    result_nodes[idx*args.avg_input_size:idx*args.avg_input_size + next_input_nodes.numel()].copy_(next_input_nodes.long())
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
                input_nodes = input_nodes_queue.get()
                # # ---------------- 3) BlockH2D ----------------
                # if args.log_time and device.type == "cuda":
                #     blocks, block_h2d_ms = time_cuda_op(
                #         lambda: [b.to(device) for b in blocks], device
                #     )
                # else:
                #     if args.log_time:
                #         t0 = time.time()
                #     blocks = [b.to(device) for b in blocks]
                #     block_h2d_ms = (time.time() - t0) * 1000.0 if args.log_time else 0.0
                idx = (it%vis)
                # ---------------- 2) FeatH2D ----------------
                def _move_feat_label():
                    idx = (it%vis)
                    if it!=0:
                        pinned_buffer[0:input_nodes.numel()].copy_(ndata[input_nodes], non_blocking=True)
                        tensor_feature_all[0:real_len[idx]][~gate_mask[idx*args.avg_input_size:idx*args.avg_input_size + real_len[idx]]].copy_(pinned_buffer[0:input_nodes.numel()], non_blocking=True)
                    while handle_dict.empty():
                        pass
                    item = handle_dict.get()
                    x = tensor_feature_all[0:real_len[idx]].float()
                    y = g.ndata["label"][output_nodes.long().cpu()].to(device)
                    y = y.long()
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
    parser.add_argument('--log_time', type=bool, default=False,help='log time')
    parser.add_argument('--sample', type=str, default="nei",
        choices=["random_walk", "nei" , "weighted"])
    parser.add_argument('--sample_ratio', type=float, default=0.01,
        help='sampling ratio for train/val/test sets (default: 0.01)')
    parser.add_argument('--vis', type=int, default=6,
        help='cache vis')
    parser.add_argument('--feat_name', type=str, default="IGB260M",
        help='feature name')
    parser.add_argument('--avg_input_size', type=int, default=350000,
        help='average input size')
    parser.add_argument('--use_uva', type=bool, default=False,
        help='use uva or not')
    parser.add_argument('--dtype', type=str, default="float32",
        help='use uva or not')

    args = parser.parse_args()

    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data...")
    dataset = IGB260MDGLDataset_part(args , load_feat=False)
    
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    g = dataset[0]
    if type(g) == list:
        g = g[0]
    print(g.ndata.keys)
    num_classes = 19
    device = torch.device('cuda', 1)
    # create GraphSAGE model
    in_size = 1024
    out_size = num_classes
    model = SAGE(in_size, 128, out_size).to(device)
    model_sage = SAGE(in_size, 256, out_size).to(device)
    model_gat = GAT(in_size, 256, out_size, num_heads=4).to(device)
    model_gcn = GCN(in_size, 256, out_size).to(device)
    model_list = [model_sage , model_gat , model_gcn]

    
    # model training
    print("Training...")
    train(args, device, g, model_list, num_classes ,args.sample)