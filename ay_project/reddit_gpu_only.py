"""Reddit fully GPU-resident training benchmark for GraphSAGE, GCN and GAT.

python -u reddit_gpu_only.py
python -u reddit_gpu_only.py --batch-size 64 --model all

GPU 0 by default. Whole graph topology, node features, labels and training
seeds reside on GPU. Mini-batch neighbor sampling is performed on GPU; this
is not full-batch training. No CPU feature extraction, H2D feature movement,
feature cache, future queue, or NUMA cache worker is used during epochs.

Per model: warm-up epoch(s), one profiled epoch, one end-to-end epoch.
The two measured epochs restore the same post-warm-up model/optimizer state.
Profiling uses synchronized wall-clock stage timers (CPU dispatch + GPU work).
Reported per-iteration stages are sampling, feature wait/preparation (GPU
feature and label gathers; there is no prefetch-worker wait), and combined
forward/loss/backward/optimizer computation.
End-to-end timing synchronizes only at epoch boundaries. Loading, graph
transfer, model creation, state restoration and warm-up are outside timing.
This is a fully-resident reference, not a proven optimal implementation.
"""

import argparse
import copy
import gc
import random
import time
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import RedditDataset
from dgl.dataloading import NeighborSampler

from models.graphsage import SAGE
from models.gcn import GCN
from models.gat import GAT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def load_reddit(device, seed):
    seed_everything(seed)
    graph = RedditDataset()[0]
    features = graph.ndata["feat"].contiguous().float().to(device)
    labels = graph.ndata["label"].reshape(-1).long().to(device)
    # Same random 60% training-seed construction as redd.py; reuse for models.
    train_idx = torch.randperm(graph.num_nodes())[:int(0.6 * graph.num_nodes())]
    for key in list(graph.ndata.keys()):
        del graph.ndata[key]
    for key in list(graph.edata.keys()):
        del graph.edata[key]
    graph = graph.int().formats(["csc"])
    graph.create_formats_()
    graph = graph.to(device)
    train_idx = train_idx.to(device=device, dtype=graph.idtype)
    torch.cuda.synchronize(device)
    return graph, features, labels, train_idx


def run_epoch(model, optimizer, graph, features, labels, train_idx,
              sampler, batch_size, device, profile=False):
    model.train()
    stages = dict(sampling=0.0, feature_wait=0.0, forward_backward=0.0)

    def stage(name, operation):
        if not profile:
            return operation()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = operation()
        torch.cuda.synchronize(device)
        stages[name] += time.perf_counter() - start
        return value

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    # Include epoch shuffle in both measured wall-clock times.
    order = torch.randperm(train_idx.numel(), device=device)
    shuffled_seeds = train_idx[order]
    batches = 0
    last_loss = None
    for offset in range(0, train_idx.numel(), batch_size):
        seeds = shuffled_seeds[offset:offset + batch_size]
        input_nodes, output_nodes, blocks = stage(
            "sampling", lambda: sampler.sample_blocks(graph, seeds))
        x, y = stage("feature_wait", lambda: (
            features[input_nodes.long()], labels[output_nodes.long()]))

        def forward_backward():
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(blocks, x), y)
            loss.backward()
            optimizer.step()
            return loss.detach()

        last_loss = stage("forward_backward", forward_backward)
        batches += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    # No .item() or per-batch logging in the measured epoch.
    loss_value = float(last_loss.item())
    if not np.isfinite(loss_value):
        raise RuntimeError("Non-finite final loss: {}".format(loss_value))
    return elapsed, stages, batches, loss_value


def benchmark_model(name, args, graph, features, labels, train_idx, device, out):
    seed_everything(args.seed)
    classes = int(labels.max().item()) + 1
    constructors = {"sage": SAGE, "gcn": GCN, "gat": GAT}
    model = constructors[name](features.shape[1], args.hidden_size, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0005)
    sampler = NeighborSampler(args.fanout)
    for epoch in range(args.warmup_epochs):
        seconds, _, _, _ = run_epoch(
            model, optimizer, graph, features, labels, train_idx,
            sampler, args.batch_size, device)
        print("{} warmup={} seconds={:.6f}".format(name, epoch, seconds), flush=True)

    # Offload snapshots to CPU, so they do not inflate measured GPU residency.
    def to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(to_cpu(v) for v in value)
        return copy.deepcopy(value)

    model_state = to_cpu(model.state_dict())
    optimizer_state = to_cpu(optimizer.state_dict())
    results = {}
    for mode in ("breakdown", "end_to_end"):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        seed_everything(args.seed + 1000)
        torch.cuda.synchronize(device)
        elapsed, stages, batches, loss = run_epoch(
            model, optimizer, graph, features, labels, train_idx,
            sampler, args.batch_size, device, profile=(mode == "breakdown"))
        results[mode] = (elapsed, stages, batches, loss)
    out.write("\nmodel={} measured_epochs=2 warmup_epochs={}\n".format(
        name, args.warmup_epochs))
    out.write("epoch_type=breakdown (separate from end_to_end epoch)\n")
    out.write("feature_wait=GPU feature/label gather and readiness; "
              "no CPU prefetch-worker wait or H2D feature transfer\n")
    out.write("stage\ttotal_ms\tavg_ms_per_batch\n")
    elapsed, stages, batches, loss = results["breakdown"]
    for stage_name, seconds in stages.items():
        out.write("{}\t{:.6f}\t{:.6f}\n".format(
            stage_name, seconds * 1000, seconds * 1000 / batches))
    out.write("SET_S_ms={:.6f} SET_E_ms={:.6f} SET_T_ms={:.6f}\n".format(
        stages["sampling"] * 1000, stages["feature_wait"] * 1000,
        stages["forward_backward"] * 1000))
    summary = (
        ("Avg SamplingOnly", "epoch_avg_sampling_ms", stages["sampling"]),
        ("Avg FeatureWait", "epoch_avg_feature_wait_ms", stages["feature_wait"]),
        ("Avg ForwardBackward", "epoch_avg_forward_backward_ms", stages["forward_backward"]),
    )
    print("{} breakdown epoch: {} batches".format(name, batches), flush=True)
    for title, field, seconds in summary:
        average_ms = seconds * 1000 / batches
        out.write("{}={:.6f}\n".format(field, average_ms))
        print("  {}: {:.6f} ms/iter".format(title, average_ms), flush=True)
    out.write("profiled_epoch_seconds={:.6f} batches={} final_loss={:.8f}\n".format(
        elapsed, batches, loss))
    seconds, _, batches, loss = results["end_to_end"]
    out.write("end_to_end_epoch_seconds={:.6f} avg_ms_per_batch={:.6f} "
              "batches_per_second={:.6f} batches={} final_loss={:.8f}\n".format(
                  seconds, seconds * 1000 / batches, batches / seconds, batches, loss))
    out.flush()
    print("{} end_to_end_epoch_seconds={:.6f}, avg_ms_per_batch={:.6f}".format(
        name, seconds, seconds * 1000 / batches), flush=True)
    del model, optimizer, model_state, optimizer_state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--model", choices=("all", "sage", "gcn", "gat"), default="all")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Default matches redd.py; use 64 for the paper's setting")
    parser.add_argument("--fanout", type=int, nargs=3, default=[5, 5, 5])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-file", default=str(Path(__file__).resolve().parent /
                                                   "reddit_gpu_only_results.txt"))
    args = parser.parse_args()
    if min(args.batch_size, args.hidden_size, args.warmup_epochs, *args.fanout) < 1:
        parser.error("batch size, hidden size, warm-up epochs and fanouts must be positive")
    if args.hidden_size % 4:
        parser.error("hidden size must be divisible by 4 for the existing GAT model")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only benchmark requires CUDA")
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    graph, features, labels, train_idx = load_reddit(device, args.seed)
    assert graph.device == device
    assert all(t.device == device for t in (features, labels, train_idx))
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write("\n=== {} Reddit GPU-only ===\n".format(datetime.now().isoformat()))
        out.write("device={} gpu_name={} nodes={} edges={} train_seeds={}\n".format(
            device, torch.cuda.get_device_name(device), graph.num_nodes(),
            graph.num_edges(), train_idx.numel()))
        out.write("batch_size={} fanout={} hidden_size={} seed={} dtype=float32\n".format(
            args.batch_size, args.fanout, args.hidden_size, args.seed))
        out.write("Full topology/features/labels/seeds on GPU; GPU neighbor sampling. "
                  "Profile=stage-synchronized wall time; E2E=epoch-boundary sync only. "
                  "Stage sum is not the normal end-to-end time.\n")
        out.flush()
        for name in (("sage", "gcn", "gat") if args.model == "all" else (args.model,)):
            benchmark_model(name, args, graph, features, labels, train_idx, device, out)
    print("Results appended to {}".format(path), flush=True)


if __name__ == "__main__":
    main()
