"""Offline K sensitivity experiment; no model training or feature transfers.

Examples (run in the project's DGL environment):
    python cache_hit_vs_k.py --dataset all
    python cache_hit_vs_k.py --dataset igb --path /lyz/data/IGB-dataset/
    python cache_hit_vs_k.py --dataset paper --paper-root /data_priv/yunzhen/ogbn-papers100M

Dataset construction, fanouts, batch sizes and hot-map presampling follow
storage_cpu_all_{igb,reddit,paper,mag}.py. Degree-only preprocessing is omitted.
The policy follows those simulators: freeze the hottest 50% of cache slots;
admit current misses reused within K future batches; evict only dynamic
entries absent from that window. This is NOT full Belady replacement.

All K values replay one sampled epoch and start with identical hot-map cache
contents. Hot-map frequencies come from a separate pass, as in the examples.
The full access trace is stored in CPU RAM (int32 IDs), not an online queue.
"""

import argparse
import gc
import random
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, NeighborSampler


K_VALUES = (1, 2, 4, 6, 8, 10, 12, 14)
CACHE_RATIOS = {
    "igb": (0.01, 0.02),
    "reddit": (0.15, 0.30),
    "paper": (0.003, 0.008),
    "mag": (0.002, 0.005),
}
SAMPLING = {
    "igb": ((5, 10, 15), 2048),
    "reddit": ((5, 5, 5), 200),
    "paper": ((5, 10, 15), 2048),
    "mag": ((15, 15, 15), 2048),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def load_graph_and_train_idx(name, args):
    """Keep the original graph; cache ratios always use graph.num_nodes()."""
    if name == "igb":
        from igb_dataloader import IGB260MDGLDataset_part

        dataset = IGB260MDGLDataset_part(args, load_feat=False)
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        original = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
        # Exactly the 1% subset in storage_cpu_all_igb.py, not 1% of all nodes.
        count = int(original.numel() * 0.01)
        train_idx = original[torch.randperm(original.numel())[:count]]
    elif name == "reddit":
        from dgl.data import RedditDataset

        graph = RedditDataset()[0]
        perm = torch.randperm(graph.num_nodes())
        train_idx = perm[:int(0.6 * graph.num_nodes())]
    elif name == "paper":
        from dgl.data import AsNodePredDataset
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root=args.paper_root),
            save_dir=args.paper_root,
            verbose=True,
        )
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        train_idx = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
    elif name == "mag":
        from ogb.lsc import MAG240MDataset

        dataset = MAG240MDataset(root=args.mag_root)
        edges = dataset.edge_index("paper", "paper")
        graph = dgl.graph(
            (torch.from_numpy(edges[0]), torch.from_numpy(edges[1])),
            num_nodes=dataset.num_papers,
        )
        train_idx = torch.from_numpy(dataset.get_idx_split("train")).long()
        valid_idx = torch.from_numpy(dataset.get_idx_split("valid")).long()
        middle = valid_idx.numel() // 2
        for key, indices in (
            ("train_mask", train_idx),
            ("val_mask", valid_idx[:middle]),
            ("test_mask", valid_idx[middle:]),
        ):
            mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            mask[indices] = True
            graph.ndata[key] = mask
    else:
        raise ValueError("Unknown dataset: " + name)
    if train_idx.numel() == 0:
        raise ValueError("The constructed training set is empty: " + name)
    return graph, train_idx


def make_loader(graph, train_idx, name):
    fanout, batch_size = SAMPLING[name]
    return DataLoader(
        graph, train_idx, NeighborSampler(list(fanout)),
        device="cpu", batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )


def collect_trace_and_hot_map(graph, train_idx, name):
    loader = make_loader(graph, train_idx, name)
    if len(loader) < 3:
        raise ValueError("Need at least 3 batches for endpoint-excluded metrics.")
    if graph.num_nodes() > np.iinfo(np.int32).max:
        raise ValueError("The int32 trace cannot represent this graph's IDs.")
    trace = []
    for step, (input_nodes, _, _) in enumerate(loader):
        trace.append(input_nodes.to(device="cpu", dtype=torch.int32).clone())
        if (step + 1) % 100 == 0:
            print("  trace batches: {}/{}".format(step + 1, len(loader)), flush=True)
    # A separate sampled pass avoids using the evaluation trace as its own
    # hot-map training data, matching the original scripts' two-pass approach.
    hot_map = torch.zeros(graph.num_nodes(), dtype=torch.long)
    for input_nodes, _, _ in loader:
        hot_map[input_nodes.long()] += 1
    return trace, hot_map


class HotMapWindowCache:
    def __init__(self, node_num, initial_nodes):
        self.capacity = initial_nodes.numel()
        self.fixed = int(self.capacity * 0.50)
        self.bucket = torch.full((node_num,), -1, dtype=torch.long)
        self.bucket[initial_nodes] = torch.arange(self.capacity, dtype=torch.long)
        self.dynamic_nodes = initial_nodes[self.fixed:].clone()

    def update(self, misses, trace, current_step, k):
        if misses.numel() == 0:
            return
        dynamic_size = self.capacity - self.fixed
        # Temporarily map misses to candidate slots following the cache slots.
        self.bucket[misses] = torch.arange(
            self.capacity, self.capacity + misses.numel(), dtype=torch.long
        )
        distance = torch.full((dynamic_size + misses.numel(),), k + 1,
                              dtype=torch.long)
        # Search exactly t+1 ... min(t+K, last batch), including the last batch.
        # Reverse traversal leaves the earliest distance for each node.
        end = min(current_step + k, len(trace) - 1)
        for future_step in range(end, current_step, -1):
            slots = self.bucket[trace[future_step].long()]
            slots = slots[slots >= self.fixed] - self.fixed
            distance[slots] = future_step - current_step
        self.bucket[misses] = -1
        
        threshold = torch.kthvalue(distance, dynamic_size).values

        less = distance < threshold
        equal = distance == threshold

        # 严格小于阈值的元素必定选中
        remaining = dynamic_size - less.sum()

        # 对阈值相等的元素按原始索引顺序选取
        equal_rank = equal.to(torch.int64).cumsum(dim=0)
        selected = less | (equal & (equal_rank <= remaining))

        # 前半段：不属于最小 m 个的元素需要换出
        drop_idx = torch.where(~selected[:dynamic_size])[0]

        # 后半段：属于最小 m 个的元素需要换入
        keep_idx = torch.where(selected[dynamic_size:])[0]

        assert drop_idx.numel() == keep_idx.numel()
        # keep_idx = torch.where(distance[dynamic_size:] <= k)[0]
        # drop_idx = torch.where(distance[:dynamic_size] > k)[0]
        count = min(keep_idx.numel(), drop_idx.numel())
        # if count == 0:
        #     return
        # if keep_idx.numel() > count:
        #     # Like the examples, prioritize candidates with nearest next use.
        #     insert_idx = torch.topk(distance[dynamic_size:], count,
        #                             largest=False, sorted=False).indices
        # else:
        #     insert_idx = keep_idx
        insert_idx = keep_idx
        evict_idx = drop_idx[:count]
        evicted = self.dynamic_nodes[evict_idx]
        candidates = misses[insert_idx]
        slots = self.bucket[evicted].clone()
        self.bucket[evicted] = -1
        self.bucket[candidates] = slots
        self.dynamic_nodes[evict_idx] = candidates

    def replay(self, trace, k):
        all_hits = all_requests = middle_hits = middle_requests = 0
        middle_batch_hit_sum = 0.0
        for step, stored_nodes in enumerate(trace):
            nodes = stored_nodes.long()
            hit_mask = self.bucket[nodes] >= 0
            hits = int(hit_mask.sum().item())
            requests = nodes.numel()
            if requests == 0:
                raise ValueError("Encountered an empty sampled batch.")
            all_hits += hits
            all_requests += requests
            if 0 < step < len(trace) - 1:
                middle_hits += hits
                middle_requests += requests
                middle_batch_hit_sum += hits / requests
            self.update(nodes[~hit_mask], trace, step, k)
        return {
            # Matches the examples' per-batch average, excluding endpoints.
            "hit_rate": middle_batch_hit_sum / (len(trace) - 2),
            "weighted_hit_rate": middle_hits / middle_requests,
            "all_weighted_hit_rate": all_hits / all_requests,
            "hits": middle_hits,
            "requests": middle_requests,
        }


def run_dataset(name, args):
    seed_everything(args.seed)
    print("Loading {} ...".format(name), flush=True)
    graph, train_idx = load_graph_and_train_idx(name, args)
    node_num = graph.num_nodes()
    train_size = train_idx.numel()
    print("  original nodes={}, train seeds={}".format(node_num, train_size),
          flush=True)
    trace, hot_map = collect_trace_and_hot_map(graph, train_idx, name)
    # Sampling is finished: retain only node-access metadata for the sweep.
    del graph, train_idx
    gc.collect()
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ratio in CACHE_RATIOS[name]:
        capacity = int(node_num * ratio)
        if capacity < 1:
            raise ValueError("Cache capacity must be positive.")
        initial_nodes = torch.topk(hot_map, capacity, largest=True).indices
        # Append separate run blocks rather than overwrite previous results.
        with path.open("a", encoding="utf-8") as out:
            out.write("\n=== {} ===\n".format(datetime.now().isoformat()))
            out.write("dataset={} seed={} initialization=hot_map\n".format(
                name, args.seed))
            out.write("original_graph_nodes={} train_seeds={} batches={}\n".format(
                node_num, train_size, len(trace)))
            out.write("cache_ratio={} cache_nodes={} fixed_nodes={}\n".format(
                ratio, capacity, int(capacity * 0.50)))
            out.write("fanout={} batch_size={} K_values={}\n".format(
                SAMPLING[name][0], SAMPLING[name][1], list(K_VALUES)))
            out.write("policy=current_miss_admission/window_absent_eviction\n")
            out.write("rates in [0,1]; hit_rate=mean per-batch hit rate, "
                      "excluding first/last batch (same as examples); "
                      "weighted_hit_rate=sum(hits)/sum(requests) on same batches; "
                      "all_weighted_hit_rate includes every batch\n")
            out.write("K\thit_rate\tmiss_rate\tweighted_hit_rate\t"
                      "all_weighted_hit_rate\thits\trequests\n")
            out.flush()
            for k in K_VALUES:
                cache = HotMapWindowCache(node_num, initial_nodes)
                result = cache.replay(trace, k)
                out.write("{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{}\t{}\n".format(
                    k, result["hit_rate"], 1.0 - result["hit_rate"],
                    result["weighted_hit_rate"], result["all_weighted_hit_rate"],
                    result["hits"], result["requests"]))
                out.flush()
                print("  {} ratio={:g} K={} hit_rate={:.6f}".format(
                    name, ratio, k, result["hit_rate"]), flush=True)
                del cache
            print("  saved: {}".format(path), flush=True)
        del initial_nodes
    del trace, hot_map
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all", "igb", "reddit", "paper", "mag"),
                        default="all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent /
                                                   "cache_hit_vs_k_results.txt"),
                        help="One TXT file for all datasets and cache ratios (append mode)")
    parser.add_argument("--path", default="/lyz/data/IGB-dataset/", help="IGB root")
    parser.add_argument("--dataset_size", choices=("large",), default="large",
                        help="IGB loader fixes the original graph to 100M nodes")
    parser.add_argument("--num_classes", type=int, choices=(19, 2983), default=19)
    parser.add_argument("--in_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--synthetic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--paper-root", default="/data_priv/yunzhen/ogbn-papers100M")
    parser.add_argument("--mag-root", default="/lyz/data")
    return parser.parse_args()


def main():
    args = parse_args()
    names = tuple(CACHE_RATIOS) if args.dataset == "all" else (args.dataset,)
    with torch.no_grad():
        for name in names:
            run_dataset(name, args)


if __name__ == "__main__":
    main()
