"""Feature-free evaluation of the window-counter policy in igb_gids.py.

python -u gids_cache_hit.py
python -u gids_cache_hit.py --dataset paper mag

Incremental uint8 clock and source queue advancement are retained. Only
misses reused in the future queue are admitted; candidates are truncated in
input order to the available zero-clock slots. All misses count in miss_rate.
No feature moves/training or BGL BFS ordering.
Default random prefill: IGB/Reddit sample original graph IDs; Paper/MAG sample
only IDs returned by one all-training-seed neighborhood sampling batch.
"""
import argparse
import gc
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from bgl_cache_hit import CACHE_RATIOS, SAMPLING, loader, seed_cpu
from dgl.dataloading import DataLoader, NeighborSampler
from bgl_structure_data import load_structure


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


def sampled_inputs(graph, train, name):
    for inputs, _, _ in loader(graph, train, name, shuffle=True):
        yield inputs.long().clone()


def capacity_base_and_pool(graph, train, name, seed):
    """Use the SAME sampled nodes for Paper/MAG capacity and random prefill."""
    if name not in ("paper", "mag"):
        return graph.num_nodes(), "original_graph_nodes", None
    seed_cpu(seed)
    sub_dataloader = DataLoader(
        graph, train, NeighborSampler(SAMPLING[name][0]), device="cpu",
        batch_size=train.numel(), shuffle=True, drop_last=False, num_workers=0,
    )
    graph_nodes, output_nodes, blocks = next(iter(sub_dataloader))
    pool = graph_nodes.long().clone()
    del graph_nodes, output_nodes, blocks, sub_dataloader
    gc.collect()
    return pool.numel(), "all_train_seeds_sampled_input_nodes", pool


def random_initial_nodes(base, pool, capacities, seed):
    """Uniform without replacement; ratios share prefixes of one permutation."""
    if any(not 0 <= c <= base for c in capacities):
        raise ValueError("Cache capacity exceeds random initialization pool")
    order = torch.randperm(base, generator=torch.Generator().manual_seed(seed))
    count = max(capacities, default=0)
    selected = order[:count] if pool is None else pool[order[:count]]
    return [selected[:c].clone() for c in capacities]


def run_dataset(name, args):
    seed_cpu(args.seed)
    print("Loading {} topology only".format(name), flush=True)
    graph, labels, train = load_structure(name, args)
    del labels
    base, kind, pool = capacity_base_and_pool(graph, train, name, args.seed + 5)
    capacities = [int(base * r) for r in CACHE_RATIOS[name]]
    initial = [np.empty(0, dtype=np.int64) for _ in capacities]
    if args.cache_init == "random":
        initial = random_initial_nodes(base, pool, capacities, args.seed + 10)
    elif args.cache_init == "hotmap":
        seed_cpu(args.seed + 10)
        hotmap = torch.zeros(graph.num_nodes(), dtype=torch.long)
        for inputs, _, _ in loader(graph, train, name, shuffle=True):
            hotmap[inputs.long()] += 1
        initial = [torch.topk(hotmap, c).indices.numpy().copy() for c in capacities]
        del hotmap
    del pool
    caches = [GIDSMetadataCache(graph.num_nodes(), c, ids, args.seed + 4000)
              for c, ids in zip(capacities, initial)]
    del initial
    stats = [dict(miss_sum=0.0, misses=0, requests=0, batches=0)
             for _ in caches]
    seed_cpu(args.seed + 3000)
    iterator = iter(sampled_inputs(graph, train, name))
    pending = deque()
    # Match get_iter -> init_clock -> append next -> popleft -> fetch.
    for _ in range(args.windows):
        try:
            pending.append(next(iterator))
        except StopIteration:
            break
    for cache in caches:
        cache.init_clock(pending)
    step = 0
    while pending:
        try:
            pending.append(next(iterator))
        except StopIteration:
            pass
        nodes = pending.popleft()
        for cache, stat in zip(caches, stats):
            try:
                hits, misses = cache.access(nodes, pending)
            except AssertionError as error:
                raise AssertionError("dataset={} cache_nodes={} batch={}: {}".format(
                    name, cache.capacity, step, error)) from error
            if step >= args.skip_batches:
                stat["miss_sum"] += misses / len(nodes)
                stat["misses"] += misses
                stat["requests"] += hits + misses
                stat["batches"] += 1
        step += 1
        if step % 100 == 0:
            print("{} GIDS batches={}".format(name, step), flush=True)
    if not stats[0]["batches"]:
        raise ValueError("No measured batches; reduce --skip-batches")
    with Path(args.output_file).open("a", encoding="utf-8") as out:
        out.write("\n=== {} dataset={} method=gids ===\n".format(datetime.now().isoformat(), name))
        out.write("original_nodes={} cache_capacity_base={} cache_capacity_base_nodes={} "
                  "windows={} cache_init={} seed={} skip_batches={} feature_loading=NONE "
                  "device=CPU order=shuffled policy=original_incremental_uint8_clock "
                  "clock_recompute=False overflow=truncate admission=future_reuse_only prefill_inverse_map=initialized\n".format(
                      graph.num_nodes(), kind, base, args.windows, args.cache_init,
                      args.seed, args.skip_batches))
        out.write("random_init_pool={} random_init_pool_nodes={} "
                  "random_init_seed={} replacement=False\n".format(
                      kind, base, args.seed + 10) if args.cache_init == "random"
                  else "random_init_pool=NOT_USED\n")
        out.write("dataset\tmethod\tcache_ratio\tcache_nodes\tmiss_rate\tweighted_miss_rate\t"
                  "measured_batches\tmiss_nodes\trequested_nodes\n")
        for ratio, capacity, stat in zip(CACHE_RATIOS[name], capacities, stats):
            mean = stat["miss_sum"] / stat["batches"]
            weighted = stat["misses"] / stat["requests"]
            out.write("{}\tgids\t{}\t{}\t{:.10f}\t{:.10f}\t{}\t{}\t{}\n".format(
                name, ratio, capacity, mean, weighted, stat["batches"],
                stat["misses"], stat["requests"]))
            print("{} ratio={} GIDS miss_rate={:.8f}".format(name, ratio, mean), flush=True)
    del caches, graph, train
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=("all", *CACHE_RATIOS), default=["all"])
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--cache-init", choices=("random", "hotmap", "empty"), default="random")
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--path", default="/lyz/data/IGB-dataset/")
    parser.add_argument("--dataset_size", choices=("large",), default="large")
    parser.add_argument("--num_classes", type=int, choices=(19, 2983), default=19)
    parser.add_argument("--in_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sample-frac", type=float, default=0.01)
    parser.add_argument("--reddit-root", default=None)
    parser.add_argument("--paper-root", default="/data_priv/yunzhen/ogbn-papers100M")
    parser.add_argument("--mag-root", default="/lyz/data")
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent / "gids_cache_hit_results.txt"))
    parser.add_argument("--worker-dataset", choices=tuple(CACHE_RATIOS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.windows < 1 or args.skip_batches < 0 or not 0 < args.sample_frac <= 1:
        parser.error("Invalid window, skipped batches or sample fraction")
    if "all" in args.dataset and args.dataset != ["all"]:
        parser.error("Use all alone")
    args.output_file = str(Path(args.output_file).resolve())
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    if args.worker_dataset:
        run_dataset(args.worker_dataset, args)
        return
    names = tuple(CACHE_RATIOS) if args.dataset == ["all"] else tuple(dict.fromkeys(args.dataset))
    failures = []
    for name in names:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--worker-dataset", name])
        if result.returncode:
            failures.append(name)
            with Path(args.output_file).open("a", encoding="utf-8") as out:
                out.write("dataset={} status=FAILED exitcode={}\n".format(name, result.returncode))
    print("Results: {}".format(args.output_file), flush=True)
    if failures:
        raise SystemExit("Dataset failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
