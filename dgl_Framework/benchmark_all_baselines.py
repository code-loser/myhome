"""Profile GraphSAGE baseline cache policies on four datasets, CUDA 1 default.

python -u benchmark_all_baselines.py
python -u benchmark_all_baselines.py --dataset paper --epochs 1

CPU sampling follows igb.py/redd.py/paper.py/mag.py, not *_gpu variants.
Policies: none, random, degree, hotmap, BGL PO+FIFO (five BFS sequences), GIDS.
BGL/GIDS run GraphSAGE, GCN and GAT; static methods run GraphSAGE.
GIDS follows igb_gids.py incremental uint8 clock, future-reuse admission and an eight-batch window.
All use REAL features; no FNSL,
synthetic zero features, single-NUMA restrictions, or custom IPC extensions.
Paper degree = sampled-subgraph in-degree; MAG = sampled-subgraph out-degree.
Subgraph construction follows the non-GPU baseline's single all-training-seed
sampling pass with its dataset fanout, NOT a full-neighbor [-1,-1,-1] sampler.
Subgraphs retain original IDs; capacity always uses the ORIGINAL graph size.

Each dataset is loaded in a separate subprocess; each ratio/policy starts a
fresh model, optimizer and cache. Random caches use distinct uniform node IDs
(the original randint approach could silently underfill the effective cache).
By default each configuration runs two consecutive epochs: end-to-end then
breakdown. --epochs counts pairs; optional warm-up is excluded. E2E only
synchronizes at epoch boundaries; breakdown synchronizes each stage.
BGL seed ordering is generated once per dataset outside training timers and
reused across all models, cache ratios and epochs.
"""

import argparse
import gc
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, NeighborSampler
from bgl_core import FIFOFeatureCache, ProximityAwareOrdering


CACHE_RATIOS = {
    "igb": (0.01, 0.02), "reddit": (0.15, 0.30),
    "paper": (0.05, 0.10), "mag": (0.03, 0.06),
}
SAMPLING = {
    "igb": ([5, 10, 15], 2048), "reddit": ([5, 5, 5], 200),
    "paper": ([5, 10, 15], 2048), "mag": ([15, 15, 15], 2048),
}
METHODS = ("none", "random", "degree", "hotmap", "bgl", "gids")
BASELINES = {"none": "DGL-no-cache", "random": "DGL-random-cache",
             "degree": "PaGraph-degree-policy", "hotmap": "GNNLab-hotmap-policy",
             "bgl": "BGL-PO-FIFO", "gids": "GIDS-window-counter"}
STAGES = ("sampling", "block_h2d", "feature_wait", "forward", "backward")


class GraphSAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList([
            dglnn.SAGEConv(in_size, hidden_size, "mean"),
            dglnn.SAGEConv(hidden_size, hidden_size, "mean"),
            dglnn.SAGEConv(hidden_size, out_size, "mean"),
        ])
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, features):
        for index, (layer, block) in enumerate(zip(self.layers, blocks)):
            features = layer(block, features)
            if index < len(self.layers) - 1:
                features = self.dropout(F.relu(features))
        return features


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def load_dataset(name, args, return_validation=False, return_test=False):
    """Keep loading and train splits consistent with non-GPU baseline files."""
    seed_everything(args.seed)
    if name == "igb":
        from igb_dataloader import IGB260MDGLDataset

        dataset = IGB260MDGLDataset(args)
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        graph = dgl.add_self_loop(graph).to("cpu")  # igb.py preprocessing
        full_train = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
        count = max(1, int(full_train.numel() * args.sample_frac))
        train_idx = full_train[torch.randperm(full_train.numel())[:count]]
        classes = args.num_classes
    elif name == "reddit":
        from dgl.data import RedditDataset

        graph = RedditDataset()[0]
        order = torch.randperm(graph.num_nodes())
        train_idx = order[:int(graph.num_nodes() * 0.6)].clone()
        classes = int(graph.ndata["label"].max().item()) + 1
    elif name == "paper":
        from dgl.data import AsNodePredDataset
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root=args.paper_root),
            save_dir=args.paper_root, verbose=True)
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        train_idx = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
        classes = 172
    elif name == "mag":
        from ogb.lsc import MAG240MDataset

        dataset = MAG240MDataset(root=args.mag_root)
        edges = dataset.edge_index("paper", "paper")
        graph = dgl.graph((torch.from_numpy(edges[0]), torch.from_numpy(edges[1])),
                          num_nodes=dataset.num_papers)
        # Match mag.py's float32 feature store, not synthetic or GPU-only data.
        graph.ndata["feat"] = torch.from_numpy(dataset.paper_feat).float()
        graph.ndata["label"] = torch.from_numpy(dataset.paper_label).long()
        train_idx = torch.from_numpy(dataset.get_idx_split("train")).long()
        classes = 153
    else:
        raise ValueError(name)
    if return_validation:
        if name == "reddit":
            val_idx = order[int(graph.num_nodes() * 0.6):int(graph.num_nodes() * 0.8)].clone()
        elif name == "mag":
            valid = torch.from_numpy(dataset.get_idx_split("valid")).long()
            val_idx = valid[:valid.numel() // 2].clone()
        else:
            val_idx = torch.nonzero(graph.ndata["val_mask"], as_tuple=True)[0]
            if name == "igb":
                count = max(1, int(val_idx.numel() * args.sample_frac))
                val_idx = val_idx[torch.randperm(val_idx.numel())[:count]]
    if return_test:
        if name == "reddit":
            # Same random 60/20/20 split as redd.py; last 20% are held out.
            test_idx = order[int(graph.num_nodes() * 0.8):].clone()
        elif name == "paper":
            test_idx = torch.nonzero(graph.ndata["test_mask"], as_tuple=True)[0]
        else:
            raise ValueError("Test accuracy is configured only for Reddit/Paper")
    features = graph.ndata["feat"]
    labels = graph.ndata["label"].reshape(-1).long()
    # Preserve the feature store separately so node_subgraph cannot duplicate
    # features or labels during degree initialization. Sampling semantics stay
    # unchanged because only the graph topology is used by NeighborSampler.
    for key in list(graph.ndata.keys()):
        del graph.ndata[key]
    for key in list(graph.edata.keys()):
        del graph.edata[key]
    if features.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("Baseline feature store and labels must be on CPU")
    if train_idx.numel() == 0:
        raise ValueError("Empty training split")
    if return_test:
        if return_validation:
            return graph, features, labels, train_idx, classes, val_idx, test_idx
        return graph, features, labels, train_idx, classes, test_idx
    if return_validation:
        return graph, features, labels, train_idx, classes, val_idx
    return graph, features, labels, train_idx, classes


def make_loader(graph, train_idx, name, shuffle=True):
    fanout, batch_size = SAMPLING[name]
    return DataLoader(graph, train_idx, NeighborSampler(fanout), device="cpu",
                      batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0)


@torch.no_grad()
def evaluate_test(model, graph, features, labels, test_idx, name, device, seed):
    """Sampled-neighbor final-model accuracy; bypass cache, outside timers."""
    if test_idx.numel() == 0:
        raise ValueError("Empty test split")
    seed_everything(seed)
    model.eval()
    correct = total = 0
    for inputs, outputs, blocks in make_loader(graph, test_idx, name, shuffle=False):
        x = features[inputs.long()].float().to(device)
        y = labels[outputs.long()].to(device)
        if bool((y < 0).any().item()):
            raise ValueError("Test split contains unavailable labels")
        predictions = model([block.to(device) for block in blocks], x).argmax(dim=1)
        correct += int((predictions == y).sum().item())
        total += outputs.numel()
    return correct / total, total


def degree_scores(graph, train_idx, name, seed):
    seed_everything(seed)
    if name not in ("paper", "mag"):
        return graph.in_degrees().long(), "original_graph_in_degree"
    # Deliberately preserve the sampled induced-subgraph policy in paper.py
    # and mag.py. No original-graph degree substitution is made here.
    fanout, _ = SAMPLING[name]
    loader = DataLoader(graph, train_idx, NeighborSampler(fanout), device="cpu",
                        batch_size=train_idx.numel(), shuffle=(name == "mag"),
                        drop_last=False, num_workers=0)
    iterator = iter(loader)
    nodes, output_nodes, blocks = next(iterator)
    del output_nodes, blocks, iterator, loader
    gc.collect()
    subgraph = dgl.node_subgraph(graph, nodes, relabel_nodes=False)
    scores = (subgraph.in_degrees() if name == "paper"
              else subgraph.out_degrees()).long()
    assert scores.numel() == graph.num_nodes()
    del subgraph, nodes
    gc.collect()
    return scores, "sampled_subgraph_{}_degree".format("in" if name == "paper" else "out")


def hotmap_scores(graph, train_idx, name, seed):
    seed_everything(seed)
    scores = torch.zeros(graph.num_nodes(), dtype=torch.long)
    loader = make_loader(graph, train_idx, name)
    for step, (input_nodes, _, _) in enumerate(loader):
        scores[input_nodes.long()] += 1
        if (step + 1) % 100 == 0:
            print("{} hotmap {}/{}".format(name, step + 1, len(loader)), flush=True)
    return scores


class BGLFeatureCache(FIFOFeatureCache):
    def fetch(self, input_nodes, output_nodes, labels):
        x, y, _, _, _ = self.fetch_and_update(input_nodes, output_nodes, labels)
        return x, y


# Local copy of the original incremental-clock metadata policy.
class GIDSMetadataCache:
    """Literal incremental-clock policy from igb_gids.py; no features."""
    def __init__(self, node_count, capacity, initial_nodes, seed):
        self.capacity = capacity
        self.cache_bucket = torch.full((node_count,), -1, dtype=torch.long)
        self.point = torch.full((capacity,), -1, dtype=torch.long)
        self.clock = torch.zeros(capacity, dtype=torch.uint8)
        self.generator = torch.Generator().manual_seed(seed)
        initial_nodes = torch.as_tensor(initial_nodes, dtype=torch.long)
        if not 0 <= capacity <= node_count or initial_nodes.numel() > capacity:
            raise ValueError("Invalid capacity/prefill")
        slots = torch.arange(initial_nodes.numel(), dtype=torch.long)
        self.cache_bucket[initial_nodes] = slots
        # Required inverse map: original prefill omits this assignment.
        self.point[slots] = initial_nodes

    def init_clock(self, input_nodes_queue):
        self.clock.zero_()
        for input_nodes in input_nodes_queue:
            mask = self.cache_bucket[input_nodes] != -1
            self.clock[self.cache_bucket[input_nodes[mask]]] += 1

    def access(self, input_nodes, input_nodes_queue):
        on_cache_mask = self.cache_bucket[input_nodes] != -1
        nids_cpu = input_nodes[~on_cache_mask]
        nids_cuda = input_nodes[on_cache_mask]
        # Admission considers future reuse; metrics still count ALL misses.
        reuse_count = torch.zeros(nids_cpu.numel(), dtype=torch.long)
        for input_nodes_temp in input_nodes_queue:
            mask = torch.isin(nids_cpu, input_nodes_temp)
            reuse_count[mask] += 1
        candidate_idx = torch.nonzero(reuse_count > 0, as_tuple=True)[0]
        evict_nodes = torch.where(self.clock == 0)[0]
        insert_num = min(candidate_idx.numel(), evict_nodes.numel())
        admitted_idx = candidate_idx[:insert_num]
        admitted = nids_cpu[admitted_idx]
        idx = torch.randperm(evict_nodes.numel(), generator=self.generator)[:insert_num]
        evict_idx = evict_nodes[idx]
        old = self.point[evict_idx]
        self.cache_bucket[old[old >= 0]] = -1
        self.cache_bucket[admitted] = evict_idx
        self.point[evict_idx] = admitted
        self.clock[self.cache_bucket[nids_cuda]] -= 1
        self.clock[evict_idx] = reuse_count[admitted_idx].to(self.clock.dtype)
        # Expose exact feature rows and slots for the real GPU cache adapter.
        self.last_admitted_miss_indices = admitted_idx
        self.last_replace = evict_idx
        return nids_cuda.numel(), nids_cpu.numel()


class GIDSFeatureCache:
    """Real feature storage using exactly GIDSMetadataCache's slot decisions."""
    windows = 8

    def begin_epoch(self):
        self.clock_init_seconds = None

    def __init__(self, features, capacity, scores, device, chunk_size, seed):
        self.features, self.device, self.capacity = features, device, capacity
        nodes = torch.topk(scores, capacity).indices
        self.metadata = GIDSMetadataCache(features.shape[0], capacity, nodes, seed)
        self.gpu_features = torch.empty((capacity, features.shape[1]),
                                        dtype=torch.float32, device=device)
        for start in range(0, capacity, chunk_size):
            self.gpu_features[start:start + chunk_size].copy_(
                features[nodes[start:start + chunk_size]].float())
        torch.cuda.synchronize(device)
        self.begin_epoch()

    def init_clock(self, pending):
        start = time.perf_counter()
        self.metadata.init_clock([batch[0].long() for batch in pending])
        self.clock_init_seconds = time.perf_counter() - start

    def fetch(self, input_nodes, output_nodes, labels, future):
        nodes = input_nodes.long()
        slots = self.metadata.cache_bucket[nodes].clone()
        hit_pos = torch.nonzero(slots >= 0, as_tuple=True)[0]
        miss_pos = torch.nonzero(slots < 0, as_tuple=True)[0]
        x = torch.empty((nodes.numel(), self.features.shape[1]),
                        dtype=torch.float32, device=self.device)
        if hit_pos.numel():
            x[hit_pos.to(self.device)] = self.gpu_features[slots[hit_pos].to(self.device)]
        missing_features = self.features[nodes[miss_pos]].float().to(self.device)
        if miss_pos.numel():
            x[miss_pos.to(self.device)] = missing_features
        # Current batch receives all miss features; only reusable candidates
        # admitted by metadata are written into persistent cache slots.
        self.metadata.access(nodes, future)
        replace = self.metadata.last_replace
        if replace.numel():
            rows = self.metadata.last_admitted_miss_indices.to(self.device)
            self.gpu_features[replace.to(self.device)] = missing_features[rows]
        return x, labels[output_nodes.long()].to(self.device)


class StaticFeatureCache:
    def __init__(self, features, node_num, capacity, method, scores, device,
                 prefill_batch_size):
        self.features, self.device = features, device
        self.bucket = self.gpu_features = None
        self.capacity = 0 if method == "none" else capacity
        if method == "none":
            return
        if method == "random":
            # Uniform without replacement: actual distinct capacity = requested C.
            nodes = torch.randperm(node_num)[:capacity].clone()
        else:
            nodes = torch.topk(scores, k=capacity, largest=True).indices
        self.bucket = torch.full((node_num,), -1, dtype=torch.long)
        self.bucket[nodes] = torch.arange(capacity, dtype=torch.long)
        self.gpu_features = torch.empty((capacity, features.shape[1]),
                                        dtype=torch.float32, device=device)
        # Bound temporary CPU and GPU tensors during potentially large prefills.
        for start in range(0, capacity, prefill_batch_size):
            end = min(start + prefill_batch_size, capacity)
            self.gpu_features[start:end].copy_(features[nodes[start:end]].float())
        torch.cuda.synchronize(device)

    def fetch(self, input_nodes, output_nodes, labels):
        input_nodes = input_nodes.long()
        output_nodes = output_nodes.long()
        if self.bucket is None:
            return (self.features[input_nodes].float().to(self.device),
                    labels[output_nodes].to(self.device))
        slots = self.bucket[input_nodes]
        hit_mask = slots >= 0
        hits = torch.nonzero(hit_mask, as_tuple=True)[0]
        misses = torch.nonzero(~hit_mask, as_tuple=True)[0]
        x = torch.empty((input_nodes.numel(), self.features.shape[1]),
                        dtype=torch.float32, device=self.device)
        if hits.numel():
            x[hits.to(self.device)] = self.gpu_features[slots[hits].to(self.device)]
        if misses.numel():
            x[misses.to(self.device)] = self.features[input_nodes[misses]].float().to(self.device)
        return x, labels[output_nodes].to(self.device)


def profile_epoch(model, optimizer, loader, cache, labels, device, profile=True):
    model.train()
    if isinstance(cache, GIDSFeatureCache):
        cache.begin_epoch()
    totals = {stage: 0.0 for stage in STAGES}

    def timed(stage, operation):
        if not profile:
            return operation()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = operation()
        torch.cuda.synchronize(device)
        totals[stage] += (time.perf_counter() - start) * 1000
        return value

    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    iterator = iter(loader)
    pending = deque()

    if isinstance(cache, GIDSFeatureCache):
        def fill_initial_window():
            for _ in range(cache.windows):
                try:
                    pending.append(next(iterator))
                except StopIteration:
                    break
        timed("sampling", fill_initial_window)
        timed("feature_wait", lambda: cache.init_clock(pending))

    def sample_batch():
        if not isinstance(cache, GIDSFeatureCache):
            return next(iterator)
        # Exact source order: initial 8 -> clock init -> append -> pop.
        try:
            pending.append(next(iterator))
        except StopIteration:
            pass
        return pending.popleft()
    batches = 0
    last_loss = None
    for _ in range(len(loader)):
        input_nodes, output_nodes, blocks = timed("sampling", sample_batch)
        blocks = timed("block_h2d", lambda: [b.to(device) for b in blocks])
        if isinstance(cache, GIDSFeatureCache):
            x, y = timed("feature_wait", lambda: cache.fetch(
                input_nodes, output_nodes, labels,
                [batch[0].long() for batch in pending]))
        else:
            x, y = timed("feature_wait", lambda: cache.fetch(input_nodes, output_nodes, labels))
        # Include loss construction in forward, parameter update in backward.
        loss = timed("forward", lambda: F.cross_entropy(model(blocks, x), y))

        def backward():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        timed("backward", backward)
        last_loss = loss.detach()
        batches += 1
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    if batches == 0:
        raise ValueError("No batches")
    loss_value = last_loss.item()  # outside timed epoch
    if not np.isfinite(loss_value):
        raise RuntimeError("Non-finite loss")
    return totals, batches, wall_seconds, loss_value


def write_result(out, name, ratio, method, epoch, totals, batches, wall, loss,
                 capacity, init_seconds, phase="measured", model_name="GraphSAGE"):
    averages = {k: value / batches for k, value in totals.items()}
    averages["forward_backward"] = averages["forward"] + averages["backward"]
    averages["extraction_total"] = averages["block_h2d"] + averages["feature_wait"]
    averages["stage_total"] = sum(averages[k] for k in STAGES)
    out.write("dataset={} cache_ratio={} method={} baseline={} model={} "
              "epoch={} phase={} effective_cache_nodes={} init_seconds={:.6f}\n".format(
                  name, ratio, method, BASELINES[method], model_name, epoch, phase, capacity, init_seconds))
    for key, value in averages.items():
        out.write("epoch_avg_{}_ms={:.6f}\n".format(key, value))
    out.write("batches={} profiled_epoch_seconds={:.6f} final_loss={:.8f}\n\n".format(
        batches, wall, loss))
    out.flush()
    print("{} ratio={} method={} {} S={:.4f} E_feat={:.4f} T={:.4f} ms/iter".format(
        name, ratio, method, phase, averages["sampling"], averages["feature_wait"],
        averages["forward_backward"]), flush=True)


def run_dataset(name, args):
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    path = Path(args.output_file)
    with path.open("a", encoding="utf-8") as out:
        out.write("\n=== {} dataset={} device={} gpu_name={} ===\n".format(
            datetime.now().isoformat(), name, device, torch.cuda.get_device_name(device)))
        out.flush()
        test_idx = None
        if name in ("reddit", "paper") and any(m in ("bgl", "gids") for m in args.methods):
            graph, features, labels, train_idx, classes, test_idx = load_dataset(
                name, args, return_test=True)
        else:
            graph, features, labels, train_idx, classes = load_dataset(name, args)
        out.write("original_nodes={} edges={} train_seeds={} feature_dim={} "
                  "source_feature_dtype={} cache_dtype=float32 fanout={} batch_size={}\n".format(
                      graph.num_nodes(), graph.num_edges(), train_idx.numel(),
                      features.shape[1], features.dtype, *SAMPLING[name]))
        # Compute policy metadata once per dataset, outside all epoch timers.
        degree, degree_source = degree_scores(graph, train_idx, name, args.seed + 11)
        hotmap_start = time.perf_counter()
        hotmap = hotmap_scores(graph, train_idx, name, args.seed + 22)
        hotmap_seconds = time.perf_counter() - hotmap_start
        out.write("shared_hotmap_presampling_seconds={:.6f}\n".format(hotmap_seconds))
        out.write("degree_source={} warmup_epochs={} measured_pairs={} "
                  "sampler=CPU feature_store=CPU model_device={}\n".format(
                      degree_source, args.warmup_epochs, args.epochs, device))
        out.write("All average times in ms/iter; synchronized wall-clock stages. "
                  "feature_wait includes CPU cache lookup/extraction, H2D and GPU assembly; "
                  "extraction_total=block_h2d+feature_wait. No validation in timed epochs. "
                  "none uses effective capacity 0 at both requested ratio settings.\n")
        out.write("capacity_base=original_graph_nodes for all methods/datasets; "
                  "BGL num_sequences=5 init=hotmap hits_do_not_refresh_FIFO; "
                  "two_consecutive_epochs_per_pair=E2E_then_breakdown; "
                  "model_optimizer_and_cache_continue_between_epochs. "
                  "GIDS windows=8 init=hotmap overflow=truncate admission=future_reuse_only clock=incremental_uint8; "
                  "BGL/GIDS models=GraphSAGE,GCN,GAT; static_models=GraphSAGE. "
                  "BGL_order_scope=once_per_dataset_reused_all_models_ratios_epochs; "
                  "shared_seed_initialization_seconds is paid once per dataset.\n")
        out.flush()
        bgl_train_idx = None
        shared_seed_initialization_seconds = 0.0
        if "bgl" in args.methods:
            print("{} BGL: generating shared seed order once (sequences=5)".format(name), flush=True)
            setup_start = time.perf_counter()
            ordering = ProximityAwareOrdering(graph, train_idx, direction="in")
            shared_order_setup_seconds = time.perf_counter() - setup_start
            order_start = time.perf_counter()
            bgl_train_idx = ordering.epoch_order(5, args.seed + 2000)
            shared_seed_order_seconds = time.perf_counter() - order_start
            shared_seed_initialization_seconds = shared_order_setup_seconds + shared_seed_order_seconds
            del ordering
            gc.collect()
            out.write("SHARED_INITIALIZATION dataset={} method=bgl num_sequences=5 "
                      "order_setup_seconds={:.6f} seed_order_seconds={:.6f} "
                      "shared_seed_initialization_seconds={:.6f} "
                      "scope=once_per_dataset reused_across=models_ratios_epochs\n".format(
                          name, shared_order_setup_seconds, shared_seed_order_seconds,
                          shared_seed_initialization_seconds))
            out.flush()
            print("{} BGL shared_seed_initialization_seconds={:.6f}".format(
                name, shared_seed_initialization_seconds), flush=True)
        for ratio in CACHE_RATIOS[name]:
            configurations = [(method, model_name) for method in args.methods
                              for model_name in (("GraphSAGE", "GCN", "GAT")
                                                 if method in ("bgl", "gids") else ("GraphSAGE",))]
            for method, model_name in configurations:
                print("Starting dataset={} ratio={} method={} model={}".format(
                    name, ratio, method, model_name), flush=True)
                model = optimizer = cache = loader = None
                try:
                    seed_everything(args.seed + 33)
                    torch.cuda.empty_cache()
                    init_start = time.perf_counter()
                    capacity = int(graph.num_nodes() * ratio)
                    if method == "gids":
                        cache = GIDSFeatureCache(features, capacity, hotmap, device,
                                                 args.prefill_batch_size, args.seed + 4000)
                    elif method == "bgl":
                        cache = BGLFeatureCache(features, capacity, device)
                        cache.prefill_hotmap(hotmap)
                        torch.cuda.synchronize(device)
                    else:
                        cache = StaticFeatureCache(
                            features, graph.num_nodes(), capacity,
                            method, degree if method == "degree" else hotmap,
                            device, args.prefill_batch_size)
                    init_seconds = time.perf_counter() - init_start
                    order_setup_seconds = 0.0
                    # Independent, identical model initialization for every policy.
                    seed_everything(args.seed + 44)
                    model_type = {"GraphSAGE": GraphSAGE, "GCN": GCN, "GAT": GAT}[model_name]
                    model = model_type(features.shape[1], 256, classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
                    loader = make_loader(graph, bgl_train_idx if method == "bgl" else train_idx,
                                         name, shuffle=(method != "bgl"))
                    aggregate = {stage: 0.0 for stage in STAGES}
                    total_batches = total_wall = 0
                    # --epochs counts measurement pairs: E2E then breakdown.
                    for epoch in range(args.warmup_epochs + 2 * args.epochs):
                        measured_index = epoch - args.warmup_epochs
                        profile = measured_index >= 0 and measured_index % 2 == 1
                        mode = "breakdown" if profile else "end-to-end"
                        order_seconds = 0.0
                        # Pair CPU sampling streams and model RNG across methods.
                        seed_everything(args.seed + 1000 + epoch)
                        totals, batches, wall, loss = profile_epoch(
                            model, optimizer, loader, cache, labels, device, profile=profile)
                        if epoch < args.warmup_epochs:
                            print("{} ratio={} method={} warmup={}".format(
                                name, ratio, method, epoch), flush=True)
                            continue
                        out.write("dataset={} cache_ratio={} method={} model={} epoch={} timing={} "
                                  "num_sequences={} cache_init_seconds={:.6f} "
                                  "order_setup_seconds={:.6f} seed_order_seconds={:.6f} "
                                  "seed_initialization_seconds={:.6f}\n".format(
                                      name, ratio, method, model_name, epoch, mode,
                                      5 if method == "bgl" else 0, init_seconds,
                                      order_setup_seconds, order_seconds,
                                      order_setup_seconds + order_seconds))
                        if method in ("bgl", "gids"):
                            clock_seconds = cache.clock_init_seconds if method == "gids" else 0.0
                            initialization = init_seconds + clock_seconds
                            out.write("shared_seed_initialization_seconds={:.6f} "
                                      "shared_seed_cost_scope=once_per_dataset "
                                      "shared_seed_cost_in_initialization_seconds=False\n".format(
                                          shared_seed_initialization_seconds if method == "bgl" else 0.0))
                            out.write("initialization dataset={} cache_ratio={} method={} model={} "
                                      "epoch={} prefill_seconds={:.6f} clock_init_seconds={:.6f} "
                                      "initialization_seconds={:.6f} "
                                      "initialization_with_hotmap_seconds={:.6f}\n".format(
                                          name, ratio, method, model_name, epoch, init_seconds,
                                          clock_seconds, initialization, initialization + hotmap_seconds))
                            out.write("initialization_scope=cache_allocation_mapping_and_prefill"
                                      "+first_window_clock_for_GIDS; BGL_seed_order_is_shared_dataset_cost; "
                                      "GIDS_clock_is_also_in_epoch_feature_wait; "
                                      "one_time_costs_are_repeated_as_reference_not_new_work\n")
                            print("{} ratio={} method={} model={} initialization_seconds={:.6f} "
                                  "prefill_seconds={:.6f} clock_init_seconds={:.6f}".format(
                                      name, ratio, method, model_name, initialization,
                                      init_seconds, clock_seconds), flush=True)
                        if not profile:
                            out.write("end_to_end_epoch_seconds={:.6f} "
                                      "epoch_with_seed_order_seconds={:.6f} batches={} "
                                      "final_loss={:.8f}\n\n".format(
                                          wall, wall + order_seconds, batches, loss))
                            out.flush()
                            print("{} ratio={} method={} end_to_end_epoch_seconds={:.6f} "
                                  "seed_order_seconds={:.6f}".format(
                                      name, ratio, method, wall, order_seconds), flush=True)
                            continue
                        write_result(out, name, ratio, method, epoch, totals, batches,
                                     wall, loss, cache.capacity, init_seconds, model_name=model_name)
                        for key in STAGES:
                            aggregate[key] += totals[key]
                        total_batches += batches
                        total_wall += wall
                    write_result(out, name, ratio, method, "all", aggregate,
                                 total_batches, total_wall, loss, cache.capacity,
                                 init_seconds, phase="summary", model_name=model_name)
                    if method in ("bgl", "gids") and test_idx is not None:
                        test_acc, tested_nodes = evaluate_test(
                            model, graph, features, labels, test_idx, name, device,
                            args.seed + 9000)
                        split = "random_last_20_percent" if name == "reddit" else "official_test_mask"
                        out.write("TEST dataset={} cache_ratio={} method={} model={} "
                                  "test_acc={:.8f} test_nodes={} test_split={} "
                                  "evaluation=sampled_neighbors model_checkpoint=final "
                                  "outside_training_timing=True cache_unchanged=True\n".format(
                                      name, ratio, method, model_name, test_acc, tested_nodes, split))
                        out.flush()
                        print("TEST dataset={} ratio={} method={} model={} test_acc={:.8f} "
                              "({:.4f}%) nodes={}".format(
                                  name, ratio, method, model_name, test_acc,
                                  test_acc * 100, tested_nodes), flush=True)
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    out.write("dataset={} cache_ratio={} method={} model={} status=OOM error={}\n".format(
                        name, ratio, method, model_name, str(error).replace("\n", " ")))
                    out.flush()
                finally:
                    del model, optimizer, cache, loader
                    gc.collect()
                    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=("all", *CACHE_RATIOS), default=["all"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Measurement pairs: each pair has one E2E epoch and one breakdown epoch")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prefill-batch-size", type=int, default=65536)
    parser.add_argument("--path", default="/lyz/data/IGB-dataset/", help="IGB root")
    parser.add_argument("--dataset_size", choices=("large",), default="large")
    parser.add_argument("--num_classes", type=int, choices=(19, 2983), default=19)
    parser.add_argument("--in_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--synthetic", type=int, choices=(0,), default=0)
    parser.add_argument("--sample-frac", dest="sample_frac", type=float, default=0.01,
                        help="IGB train-mask subset fraction, matching igb.py")
    parser.add_argument("--paper-root", default="/data_priv/yunzhen/ogbn-papers100M")
    parser.add_argument("--mag-root", default="/lyz/data")
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent /
                                                   "baseline_breakdown_results.txt"))
    parser.add_argument("--worker-dataset", choices=tuple(CACHE_RATIOS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if "all" in args.dataset and len(args.dataset) != 1:
        parser.error("Use all alone, or specify dataset names such as paper mag")
    if min(args.epochs, args.prefill_batch_size) < 1 or args.warmup_epochs < 0:
        parser.error("epochs/prefill-batch-size must be positive; warmup-epochs must be nonnegative")
    if not 0 < args.sample_frac <= 1:
        parser.error("sample-frac must be in (0,1]")
    args.output_file = str(Path(args.output_file).resolve())
    return args


def main():
    args = parse_args()
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_dataset:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; requested cuda:{}".format(args.gpu))
        run_dataset(args.worker_dataset, args)
        return
    # Sequential subprocesses release native graph storage, CPU feature stores
    # and CUDA contexts completely between large datasets.
    names = tuple(CACHE_RATIOS) if args.dataset == ["all"] else tuple(dict.fromkeys(args.dataset))
    failed = []
    for name in names:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *sys.argv[1:], "--worker-dataset", name,
                   "--output-file", args.output_file]
        result = subprocess.run(command)
        if result.returncode:
            failed.append(name)
            with path.open("a", encoding="utf-8") as out:
                out.write("dataset={} status=FAILED exitcode={}\n".format(name, result.returncode))
    print("Results: {}".format(path), flush=True)
    if failed:
        raise SystemExit("Datasets failed (see console/TXT): " + ", ".join(failed))


if __name__ == "__main__":
    main()
