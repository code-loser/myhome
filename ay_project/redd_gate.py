import argparse
import torch.multiprocessing as mp
import multiprocessing
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
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
import sys
sys.path.append('/ay_project/cuda_ipc')
import cuda_ipc_open
import cuda_ipc_get
import cuda_ipc_close
import base64
import cupy as cp
import os
from dgl.data import RedditDataset
from queue import Queue
import storage_cpu3
from models.gat import GAT
from models.gcn import GCN
import psutil


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

def train(args, device, g, model_list, num_classes): 

    def get_iter(visibility):
        for i in range(args.vis):
            input_nodes, output_nodes, blocks = next(dataloader_iter)
            result_nodes[i*args.avg_input_size:i*args.avg_input_size + input_nodes.numel()] = input_nodes
            offset_list[i] = input_nodes.numel()
            blocks_queue.put(blocks)
            output_nodes_queue.put(output_nodes)
    
    sampler = NeighborSampler(
        [5 ,5, 5],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_labels=["label"],
    )
    idx = torch.randperm(num_nodes)
    train_idx = idx[:int(0.6 * num_nodes)]
    val_idx = idx[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    train_idx_cuda = train_idx.int().cuda()
    val_idx_cuda = val_idx.int().cuda()
    test_idx = idx[int(0.8 * num_nodes):]
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=200,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # use_uva=use_uva,
    )
    hotmap_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device="cpu",
        batch_size=200,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device="cpu",
        batch_size=200,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # use_uva=use_uva,
    )
    covered_nodes = train_idx.clone()
    seed = train_idx.clone()
    for hop in range(1):
        train_neighbors = g.in_edges(seed)[0].unique()
        covered_nodes = torch.cat([covered_nodes, train_neighbors]).unique()
        seed = train_neighbors
    degrees = torch.full((2 * g.num_nodes(),), fill_value=-1, dtype=torch.int64)
    dataloader_iter_pre = iter(hotmap_dataloader)
    for i in range(len(hotmap_dataloader)):
        input_nodes, output_nodes, blocks = next(dataloader_iter_pre)
        degrees[input_nodes] += 1
    torch.cuda.set_device(1)
    # # args.limits = len(train_dataloader)
    
    mp.set_start_method(method='forkserver', force=True)
    handle_dict = multiprocessing.Queue()
    task_queue = multiprocessing.Queue()
    signal = multiprocessing.Queue()
    p = mp.Process(target=storage_cpu3.cache_worker_main, args=(args , g.num_nodes(), 1 , handle_dict, task_queue, signal))
    p.start()
    degrees.share_memory_()
    task_queue.put(degrees)
    p = psutil.Process()
    cores_socket_0 = list(range(0,20))
    try:
        p.cpu_affinity(cores_socket_0)
    except Exception as e:
        print(f"❌ Failed to set CPU affinity: {e}")
    torch.set_num_threads(12)
    num_threads = "12"
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    start_time = time.time()
      
    while handle_dict.empty():
        pass
    item = handle_dict.get()
    init_time_start = time.time()
    handle_1 ,  shape1 = item
    blocks_queue = Queue()
    output_nodes_queue = Queue()
    handle_feature = base64.b64decode(handle_1)  # bytes
    tensor_feature_all = cuda_ipc_open.open_cuda_ipc_handle(handle_feature, list(shape1), 1 , "float32")
    vis = args.vis + 1
    result_nodes = torch.empty((vis+1)*args.avg_input_size, dtype=torch.long)
    result_nodes.share_memory_()
    offset_list = torch.empty((vis+1),dtype=torch.long)
    offset_list.share_memory_()
    task_queue.put((result_nodes,offset_list), block=False)
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
        iter_log_path = os.path.join(save_dir, "redd-log.txt")
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
                    blocks = [b.to(device) for b in blocks]

                
                # ---------------- 2) FeatH2D ----------------
                def _move_feat_label():
                    idx = (it%vis)
                    while handle_dict.empty():
                        pass
                    item = handle_dict.get()
                    x = tensor_feature_all[0:offset_list[idx]].float()
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

            val_acc = evaluate(model, g, val_dataloader, num_classes, device, use_bf16)

            epoch_tag = "warm-up" if epoch == 0 else "measured"
            print(
                f"Epoch {epoch:03d} ({epoch_tag}) | "
                f"Loss {total_loss / max(1, num_iters):.6f} | "
                f"ValAcc {val_acc:.6f}"
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
                    f.write(f"val_acc: {val_acc:.6f}\n")
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
    parser.add_argument('--sample_ratio', type=float, default=0.6,
        help='sampling ratio for train/val/test sets (default: 0.01)')
    parser.add_argument('--vis', type=int, default=6,
        help='cache vis')
    parser.add_argument('--feat_name', type=str, default="re",
        help='feature name')
    parser.add_argument('--avg_input_size', type=int, default=50000,
        help='average input size')
    parser.add_argument('--use_uva', type=bool, default=False,
        help='use uva or not')
    parser.add_argument('--dtype', type=str, default="float32",
        help='use uva or not')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data...")
    data = RedditDataset()
    g = data[0]
    g.ndata['feat'] = g.ndata['feat'].contiguous()
    g.ndata['label'] = g.ndata['label'].contiguous()
    
    num_nodes = g.num_nodes()
    idx = torch.randperm(num_nodes)
    train_idx = idx[:int(0.6 * num_nodes)]
    val_idx = idx[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    test_idx = idx[int(0.8 * num_nodes):]
    num_classes = g.ndata['label'].max().item() + 1
    dataset = CustomDataset(g, train_idx, val_idx, test_idx, num_classes)
    device = torch.device('cuda', 1)

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model_sage = SAGE(in_size, 256, out_size).to(device)
    model_gat = GAT(in_size, 256, out_size, num_heads=4).to(device)
    model_gcn = GCN(in_size, 256, out_size).to(device)
    model_list = [model_sage , model_gat , model_gcn]
    

    # model training
    print("Training...")
    train(args, device, g, model_list, num_classes)

    # test the model
    print("Testing...")
    test_idx = dataset.test_idx.to(device)
    

    test_sampler = NeighborSampler(
        [5, 10, 15],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    test_dataloader = DataLoader(
        g,
        test_idx,
        test_sampler,
        device=device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva = True
    )