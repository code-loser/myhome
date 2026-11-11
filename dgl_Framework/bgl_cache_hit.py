"""CPU-only BGL cache-size sweep with five BFS sequences; no feature loading.

python -u bgl_cache_hit.py
python -u bgl_cache_hit.py --dataset paper mag

One sampled epoch with five sequences. FIFO metadata only; no GPU cache,
feature vectors, models or training. All cache capacities see the
same sampled batches. All configurations start from the same separate
hot-map initialization (or empty cache if requested). Every batch is counted
unless explicitly excluded with --skip-batches.
"""

import argparse
import gc
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, NeighborSampler

from bgl_core import ProximityAwareOrdering, label_tv_error, round_robin_unique
from bgl_structure_data import load_structure


CACHE_RATIOS = {
    # PreGNNFlow-DATA-revised.xlsx, first worksheet, rows 4/12/20/28.
    # Paper/MAG: relative to sampled input-node count from one all-seed batch.
    # IGB/Reddit: relative to original graph node count.
    "igb": (0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.02,
            0.024, 0.028, 0.032, 0.036, 0.04, 0.044),
    "reddit": (0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
               0.3, 0.325, 0.35, 0.375, 0.4, 0.425),
    "paper": (0.045, 0.06, 0.075, 0.085, 0.105, 0.12, 0.135,
              0.15, 0.165, 0.18, 0.195, 0.21, 0.225),
    "mag": (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06,
            0.065, 0.07, 0.075, 0.08, 0.085, 0.09),
}
SAMPLING = {
    "igb": ([5, 10, 15], 2048), "reddit": ([5, 5, 5], 200),
    "paper": ([5, 10, 15], 2048), "mag": ([15, 15, 15], 2048),
}
SEQUENCES = (5,)


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


class FIFOMetadataCache:
    """Node membership + FIFO ring, equivalent to bgl_core's feature cache.

    There is no feature storage and no need for slot lookup: only membership
    affects hit statistics. DGL sampled input-node lists contain unique IDs.
    """
    def __init__(self, node_count, capacity, initial_nodes=None):
        if not 0 <= capacity <= node_count:
            raise ValueError("Invalid capacity")
        self.capacity = capacity
        self.resident = np.zeros(node_count, dtype=np.bool_)
        self.ring = np.full(capacity, -1, dtype=np.int64)
        self.head = self.size = 0
        if initial_nodes is not None:
            self.enqueue(np.asarray(initial_nodes, dtype=np.int64))

    def enqueue(self, nodes):
        count = len(nodes)
        if not count or not self.capacity:
            return
        if count >= self.capacity:
            self.resident[self.ring[self.ring >= 0]] = False
            nodes = nodes[-self.capacity:]
            self.ring[:] = nodes
            self.resident[nodes] = True
            self.head, self.size = 0, self.capacity
            return
        evictions = max(0, self.size + count - self.capacity)
        if evictions:
            slots = (np.arange(evictions) + self.head) % self.capacity
            self.resident[self.ring[slots]] = False
            self.head = (self.head + evictions) % self.capacity
            self.size -= evictions
        tail = (self.head + self.size) % self.capacity
        slots = (np.arange(count) + tail) % self.capacity
        self.ring[slots] = nodes
        self.resident[nodes] = True
        self.size += count

    def access(self, nodes):
        misses = nodes[~self.resident[nodes]]
        hits = len(nodes) - len(misses)
        # Count against the pre-update cache; hits never refresh position.
        self.enqueue(misses)
        return hits, len(misses)

    def fifo_node_ids(self):
        if not self.size:
            return np.empty(0, dtype=np.int64)
        return self.ring[(np.arange(self.size) + self.head) % self.capacity].copy()


def loader(graph, seeds, name, shuffle=False):
    fanout, size = SAMPLING[name]
    return DataLoader(graph, seeds, NeighborSampler(fanout), device="cpu",
                      batch_size=size, shuffle=shuffle, drop_last=False,
                      num_workers=0)


def cache_capacity_base(graph, train, name, seed):
    """Sample once per dataset so every ratio uses the same denominator."""
    if name not in ("paper", "mag"):
        return graph.num_nodes(), "original_graph_nodes"
    seed_cpu(seed)
    sampler1 = NeighborSampler(SAMPLING[name][0])
    sub_dataloader = DataLoader(
        graph, train, sampler1, device="cpu", batch_size=train.numel(),
        shuffle=True, drop_last=False, num_workers=0,
    )
    graph_nodes, graph_node, graph_blocks = next(iter(sub_dataloader))
    base_nodes = graph_nodes.numel()
    # Only the input-node count is needed; no induced graph is constructed.
    del graph_nodes, graph_node, graph_blocks, sub_dataloader, sampler1
    gc.collect()
    return base_nodes, "all_train_seeds_sampled_input_nodes"


def run_dataset(name, args):
    seed_cpu(args.seed)
    print("Loading {} topology/labels only".format(name), flush=True)
    graph, labels, train = load_structure(name, args)
    path = Path(args.output_file)
    with path.open("a", encoding="utf-8") as out:
        out.write("\n=== {} dataset={} BGL PO+FIFO hit-rate simulation ===\n".format(
            datetime.now().isoformat(), name))
        out.write("nodes={} train_seeds={} fanout={} batch_size={} seed={} "
                  "cache_init={} skip_batches={} feature_loading=NONE device=CPU\n".format(
                      graph.num_nodes(), train.numel(), *SAMPLING[name], args.seed,
                      args.cache_init, args.skip_batches))
        out.flush()
        ratios = CACHE_RATIOS[name]
        base_nodes, base_kind = cache_capacity_base(graph, train, name, args.seed + 5)
        capacities = [int(base_nodes * r) for r in ratios]
        out.write("cache_capacity_base={} cache_capacity_base_nodes={} "
                  "capacity_sampling_seed={} formula=int(base_nodes*cache_ratio)\n".format(
                      base_kind, base_nodes, args.seed + 5))
        out.flush()
        print("{} cache capacity base: {} nodes ({})".format(
            name, base_nodes, base_kind), flush=True)
        initial = [None] * len(capacities)
        if args.cache_init == "hotmap":
            seed_cpu(args.seed + 10)
            hotmap = torch.zeros(graph.num_nodes(), dtype=torch.long)
            for inputs, _, _ in loader(graph, train, name, shuffle=True):
                hotmap[inputs.long()] += 1
            # Cache starts with hottest entries first, as in prefill_hotmap.
            initial = [torch.topk(hotmap, c, largest=True).indices.numpy().copy()
                       if c else np.empty(0, dtype=np.int64) for c in capacities]
            del hotmap
        ordering = ProximityAwareOrdering(graph, train, args.bfs_direction)
        # Pair the same BFS roots and shifts across sequence counts. Build
        # max(m) once rather than regenerate its prefix for every configuration.
        sequences = []
        for index in range(max(args.sequences)):
            print("{} BFS sequence {}/{}".format(name, index + 1, max(args.sequences)), flush=True)
            sequences.append(ordering.bfs_sequence(np.random.RandomState(args.seed + 1000 + index)))
        shifts = np.random.RandomState(args.seed + 2000).randint(
            train.numel(), size=max(args.sequences))
        out.write("miss_rate=mean_batch_miss_rate; weighted_miss_rate=total_misses/total_requests; "
                  "capacity_source=PreGNNFlow-DATA-revised.xlsx:first_worksheet\n")
        out.write("dataset\tnum_sequences\tcache_ratio\tcache_nodes\thit_rate\tmiss_rate\t"
                  "weighted_hit_rate\tweighted_miss_rate\tmeasured_batches\t"
                  "hit_nodes\trequested_nodes\tmean_label_TV\n")
        out.flush()
        for count in args.sequences:
            order = torch.from_numpy(round_robin_unique(sequences[:count], shifts[:count]))
            tv = label_tv_error(order.numpy(), labels.numpy(), train.numpy(), SAMPLING[name][1])
            caches = [FIFOMetadataCache(graph.num_nodes(), c, ids)
                      for c, ids in zip(capacities, initial)]
            stats = [dict(hit_sum=0.0, hits=0, requests=0, batches=0) for _ in caches]
            seed_cpu(args.seed + 3000)
            batches = loader(graph, order, name, shuffle=False)
            if args.skip_batches >= len(batches):
                raise ValueError("skip-batches excludes the entire epoch")
            for step, (inputs, _, _) in enumerate(batches):
                nodes = inputs.long().numpy()
                for cache, stat in zip(caches, stats):
                    hits, misses = cache.access(nodes)
                    if step >= args.skip_batches:
                        stat["hit_sum"] += hits / len(nodes)
                        stat["hits"] += hits
                        stat["requests"] += hits + misses
                        stat["batches"] += 1
                if (step + 1) % 100 == 0:
                    print("{} sequences={} batch={}/{}".format(name, count, step + 1, len(batches)), flush=True)
            for ratio, capacity, stat in zip(ratios, capacities, stats):
                mean = stat["hit_sum"] / stat["batches"]
                weighted = stat["hits"] / stat["requests"]
                out.write("{}\t{}\t{}\t{}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{}\t{}\t{}\t{:.8f}\n".format(
                    name, count, ratio, capacity, mean, 1 - mean, weighted, 1 - weighted,
                    stat["batches"], stat["hits"], stat["requests"], tv))
                out.flush()
                print("{} ratio={} sequences={} hit_rate={:.6f} miss_rate={:.6f}".format(
                    name, ratio, count, mean, 1 - mean), flush=True)
            del caches, batches, order
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=("all", *CACHE_RATIOS), default=["all"])
    parser.add_argument("--sequences", nargs="+", type=int, choices=(5,), default=list(SEQUENCES),
                        help="Fixed to 5 (only --sequences 5 is accepted)")
    parser.add_argument("--cache-init", choices=("hotmap", "empty"), default="hotmap")
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--bfs-direction", choices=("in", "out"), default="in")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--path", default="/lyz/data/IGB-dataset/")
    parser.add_argument("--dataset_size", choices=("large",), default="large")
    parser.add_argument("--num_classes", type=int, choices=(19, 2983), default=19)
    parser.add_argument("--in_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sample-frac", dest="sample_frac", type=float, default=0.01)
    parser.add_argument("--reddit-root", default=None, help="Directory containing the two raw Reddit NPZ files")
    parser.add_argument("--paper-root", default="/data_priv/yunzhen/ogbn-papers100M")
    parser.add_argument("--mag-root", default="/lyz/data")
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent / "bgl_cache_hit_results.txt"))
    parser.add_argument("--worker-dataset", choices=tuple(CACHE_RATIOS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sequences) < 1 or args.skip_batches < 0 or not 0 < args.sample_frac <= 1:
        parser.error("Invalid sequence counts, skip-batches or sample fraction")
    if "all" in args.dataset and len(args.dataset) != 1:
        parser.error("Use all alone or explicit dataset names")
    args.sequences = sorted(set(args.sequences))
    args.output_file = str(Path(args.output_file).resolve())
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    if args.worker_dataset:
        run_dataset(args.worker_dataset, args)
        return
    names = tuple(CACHE_RATIOS) if args.dataset == ["all"] else tuple(dict.fromkeys(args.dataset))
    failures = []
    for name in names:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--worker-dataset", name,
                                 "--output-file", args.output_file])
        if result.returncode:
            failures.append(name)
            with Path(args.output_file).open("a", encoding="utf-8") as out:
                out.write("dataset={} status=FAILED exitcode={}\n".format(name, result.returncode))
    print("Results: {}".format(args.output_file), flush=True)
    if failures:
        raise SystemExit("Dataset failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
