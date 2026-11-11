"""Measure sampled-block queue storage versus K; do not train a model.

Run in the same CUDA/DGL environment as the training scripts:
    python queue_memory_vs_k.py --dataset all --gpu 1
    python queue_memory_vs_k.py --dataset reddit
    python queue_memory_vs_k.py --dataset igb --max-batches 100

Default K: 1,2,4,6,8,10,12,14. One append-only TXT contains all datasets.
Dataset loading/train seeds/fanouts/batch sizes reuse cache_hit_vs_k.py.
IGB/MAG build the same full-three-hop, original-ID subgraph used for GPU
sampling in *_gpu_gate.py; Paper follows paper_gate.py (int64 IDs).
Reddit follows redd_gate.py: CPU sampling and CPU-resident queued blocks.

The loop fills K batches, then enqueues before dequeueing, matching the
training entries' transient K+1 occupancy. Snapshot statistics are taken
immediately before every dequeue, including startup and tail draining.
Means are snapshot means, NOT time-weighted means or full-training peaks.

Primary measurements count unique tensor backing storage, including all
already-created COO/CSR/CSC formats and block node/edge data. Shared storage
is deduplicated within/across queues. This is payload storage, not Python
object overhead, allocator rounding, or CUDA context memory. Reading only
already-created formats avoids materializing new formats for measurement.

No feature cache, model, optimizer, hot/cold workers or feature prefetcher is
created. Cold-input queue storage is therefore not measured. CPU ring-buffer
capacity is estimated separately from the original allocation expressions.
Torch allocator diagnostics may omit DGL-managed graph allocations; they
must not be presented as exact total queue VRAM. Topology residency is a
K-independent baseline. An OOM is recorded, not treated as a numeric result.
"""

import argparse
import gc
from datetime import datetime
from pathlib import Path
from queue import Queue

import dgl
import torch
from dgl.dataloading import DataLoader, NeighborSampler

from cache_hit_vs_k import (
    K_VALUES, SAMPLING, load_graph_and_train_idx, seed_everything,
)


MIB = 1024 ** 2


def build_sampling_graph(graph, train_idx, name, device):
    # Original GPU entries delete graph attributes before moving topology.
    # Labels/features are not needed for this structural-memory experiment.
    for key in list(graph.ndata.keys()):
        del graph.ndata[key]
    for key in list(graph.edata.keys()):
        del graph.edata[key]
    if device.type == "cpu":
        return graph, train_idx.to(dtype=graph.idtype, device="cpu")
    full_loader = DataLoader(
        graph, train_idx, NeighborSampler([-1, -1, -1]),
        device="cpu", batch_size=train_idx.numel(), shuffle=False,
        drop_last=False, num_workers=0,
    )
    sampled_nodes, _, full_blocks = next(iter(full_loader))
    subgraph = dgl.node_subgraph(graph, sampled_nodes, relabel_nodes=False)
    del full_blocks, full_loader, sampled_nodes
    # node_subgraph can attach original-ID metadata; production entries have
    # this metadata too. No original feature tensors were copied above.
    if name in ("igb", "mag"):
        subgraph = subgraph.int()
    subgraph = subgraph.formats(["csc"])
    subgraph.create_formats_()
    subgraph = subgraph.to(device)
    seeds = train_idx.to(device=device, dtype=subgraph.idtype)
    torch.cuda.synchronize(device)
    return subgraph, seeds


def storage_record(tensor):
    if tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    # Include device in the key: CPU and CUDA pointers can have same value.
    key = (tensor.device.type, tensor.device.index, storage.data_ptr())
    return key, storage.nbytes()


def add_tensor(records, tensor):
    record = storage_record(tensor)
    if record is not None:
        key, size = record
        records[key] = max(records.get(key, 0), size)


def block_storage(blocks_queue):
    records = {}
    # This script has one producer/consumer thread, so queue contents are stable.
    for blocks in blocks_queue.queue:
        for block in blocks:
            for fmt in block.formats()["created"]:
                for tensor in block.adj_tensors(fmt):
                    add_tensor(records, tensor)
            for frame in (block.srcdata, block.dstdata, block.edata):
                # DGL data-view .values() can expose internal Column objects.
                # Public keyed access materializes the actual feature tensor.
                for key in frame.keys():
                    add_tensor(records, frame[key])
    return records


def output_storage(output_queue):
    records = {}
    for nodes in output_queue.queue:
        add_tensor(records, nodes)
    return records


def merge_records(left, right):
    merged = dict(left)
    for key, size in right.items():
        merged[key] = max(merged.get(key, 0), size)
    return merged


def device_bytes(records, kind):
    return sum(size for key, size in records.items() if key[0] == kind)


def measure_queue(graph, seeds, name, device, k, args):
    # Recreate the same stream for every K, without retaining an epoch of blocks.
    seed_everything(args.seed + 1000)
    fanout, batch_size = SAMPLING[name]
    loader = DataLoader(
        graph, seeds, NeighborSampler(list(fanout)), device=device,
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    limit = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    if limit < 1:
        raise ValueError("No sampling batches available")
    iterator = iter(loader)
    blocks_queue, output_queue = Queue(), Queue()
    produced = consumed = 0
    metrics = {}
    max_depth = 0
    max_input_nodes = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

    def enqueue():
        nonlocal produced, max_input_nodes
        input_nodes, output_nodes, blocks = next(iterator)
        max_input_nodes = max(max_input_nodes, input_nodes.numel())
        blocks_queue.put(blocks)
        output_queue.put(output_nodes)
        produced += 1
        # Local sampled tensor references die here; only queue ownership remains.

    def snapshot():
        nonlocal max_depth
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        max_depth = max(max_depth, blocks_queue.qsize())
        blocks = block_storage(blocks_queue)
        outputs = output_storage(output_queue)
        combined = merge_records(blocks, outputs)
        for prefix, records in (("blocks", blocks), ("outputs", outputs),
                                ("combined", combined)):
            for kind in ("cuda", "cpu"):
                value = device_bytes(records, kind)
                key = prefix + "_" + kind
                total, peak = metrics.get(key, (0, 0))
                metrics[key] = (total + value, max(peak, value))

    for _ in range(min(k, limit)):
        enqueue()
    while consumed < limit:
        if produced < limit:
            enqueue()
        snapshot()
        blocks_queue.get()
        output_queue.get()
        consumed += 1
        if consumed % 100 == 0:
            print("  {} K={} consumed={}/{}".format(name, k, consumed, limit),
                  flush=True)
    result = {"batches": consumed, "peak_queue_batches": max_depth,
              "max_input_nodes": max_input_nodes}
    for key, (total, peak) in metrics.items():
        result[key + "_mean_MiB"] = total / consumed / MIB
        result[key + "_peak_MiB"] = peak / MIB
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["torch_allocated_peak_delta_MiB"] = max(
            0, torch.cuda.max_memory_allocated(device) - baseline_allocated) / MIB
        result["torch_reserved_peak_delta_MiB"] = max(
            0, torch.cuda.max_memory_reserved(device) - baseline_reserved) / MIB
    else:
        result["torch_allocated_peak_delta_MiB"] = 0.0
        result["torch_reserved_peak_delta_MiB"] = 0.0
    return result


def cpu_ring_capacity(name, k, slot_nodes):
    """Allocation formulas from each gate entry; estimates, not measurements."""
    if name == "reddit":
        # vis=K+1; result_nodes=(vis+1)*B; offsets=vis+1; no gate_mask.
        return (k + 2) * slot_nodes * 8 + (k + 2) * 8
    # result_nodes int64 + gate_mask bool + real_len/offsets int64.
    offsets = k + 2 if name == "paper" else k + 1
    return (k + 1) * slot_nodes * 9 + ((k + 1) + offsets) * 8


COLUMNS = [
    "K", "status", "batches", "peak_queue_batches",
    "blocks_cuda_mean_MiB", "blocks_cuda_peak_MiB",
    "blocks_cpu_mean_MiB", "blocks_cpu_peak_MiB",
    "outputs_cuda_peak_MiB", "outputs_cpu_peak_MiB",
    "combined_cuda_peak_MiB", "combined_cpu_peak_MiB",
    "combined_cuda_peak_extra_vs_K1_MiB", "max_input_nodes",
    "cpu_ring_capacity_estimate_MiB",
    "torch_allocated_peak_delta_MiB", "torch_reserved_peak_delta_MiB",
]


def run_dataset(name, args, out):
    seed_everything(args.seed)
    device = torch.device("cpu" if name == "reddit" else "cuda:{}".format(args.gpu))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the original GPU sampling path")
        torch.cuda.set_device(device)
    out.write("\n=== {} dataset={} ===\n".format(datetime.now().isoformat(), name))
    out.write("sampling_device={} seed={} K_values={} max_batches={}\n".format(
        device, args.seed, list(K_VALUES), args.max_batches or "full_epoch"))
    out.flush()
    graph, train_idx = load_graph_and_train_idx(name, args)
    original_nodes, seed_count = graph.num_nodes(), train_idx.numel()
    sampling_graph, seeds = build_sampling_graph(graph, train_idx, name, device)
    del graph, train_idx
    gc.collect()
    slot_nodes = args.reddit_slot_nodes if name == "reddit" else args.slot_nodes
    out.write("original_graph_nodes={} train_seeds={} topology_idtype={}\n".format(
        original_nodes, seed_count, sampling_graph.idtype))
    out.write("fanout={} batch_size={} slot_nodes={}\n".format(
        SAMPLING[name][0], SAMPLING[name][1], slot_nodes))
    out.write("\t".join(COLUMNS) + "\n")
    out.flush()
    reference = None
    for k in K_VALUES:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            result = measure_queue(sampling_graph, seeds, name, device, k, args)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            out.write("{}\tOOM\t{}\n".format(k, "\t".join(["NA"] * (len(COLUMNS)-2))))
            out.flush()
            print("{} K={}: OOM".format(name, k), flush=True)
            continue
        if k == 1:
            reference = result["combined_cuda_peak_MiB"]
        result.update(K=k, status="ok")
        result["combined_cuda_peak_extra_vs_K1_MiB"] = (
            result["combined_cuda_peak_MiB"] - reference
            if reference is not None else "NA")
        result["cpu_ring_capacity_estimate_MiB"] = cpu_ring_capacity(
            name, k, slot_nodes) / MIB
        row = ["{:.6f}".format(result[c]) if isinstance(result[c], float)
               else str(result[c]) for c in COLUMNS]
        out.write("\t".join(row) + "\n")
        out.flush()
        print("{} K={} depth={} blocks_GPU_peak={:.3f} MiB".format(
            name, k, result["peak_queue_batches"], result["blocks_cuda_peak_MiB"]),
              flush=True)
    del sampling_graph, seeds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all", "igb", "reddit", "paper", "mag"),
                        default="all")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0: full epoch; positive: explicitly truncated measurement")
    parser.add_argument("--slot-nodes", type=int, default=350000,
                        help="CPU ring-buffer slot capacity for IGB/Paper/MAG")
    parser.add_argument("--reddit-slot-nodes", type=int, default=50000)
    parser.add_argument("--path", default="/lyz/data/IGB-dataset/")
    parser.add_argument("--dataset_size", choices=("large",), default="large")
    parser.add_argument("--num_classes", type=int, choices=(19, 2983), default=19)
    parser.add_argument("--in_memory", type=int, choices=(0, 1), default=1)
    parser.add_argument("--synthetic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--paper-root", default="/data_priv/yunzhen/ogbn-papers100M")
    parser.add_argument("--mag-root", default="/lyz/data")
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent /
                                                   "queue_memory_vs_k_results.txt"))
    args = parser.parse_args()
    if args.max_batches < 0 or min(args.slot_nodes, args.reddit_slot_nodes) < 1:
        parser.error("max-batches must be >= 0 and slot capacities must be positive")
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out, torch.no_grad():
        out.write("\nQueue payload experiment; units=MiB (2^20 bytes).\n")
        out.write("Statistics: pre-dequeue snapshots, including startup/tail; "
                  "enqueue-before-dequeue gives up to K+1 batches.\n")
        out.write("combined deduplicates shared block/output storage; "
                  "CPU rings are formula estimates; cold-input queue excluded.\n")
        out.write("Torch allocator columns are diagnostics and can omit DGL "
                  "allocations. This is queue overhead, not unused-feature waste.\n")
        names = ("igb", "reddit", "paper", "mag") if args.dataset == "all" else (args.dataset,)
        for name in names:
            run_dataset(name, args, out)


if __name__ == "__main__":
    main()
