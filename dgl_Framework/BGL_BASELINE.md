# Multi-dataset BGL-style baseline

Reference: supplied BGL.pdf / [arXiv:2112.08541v1](https://arxiv.org/pdf/2112.08541),
sections 3.2.1 and 3.2.2, pp.5--6. The paper calls the policy **FIFO**, not LRU.
This entry implements proximity-aware ordering plus the requested FIFO cache
inside a redd.py-style single-GPU training loop, extended to four datasets.
It does not claim to reproduce
distributed graph partitioning, inter-GPU sharing or the full BGL engine.

## Run

Copy `redd_bgl.py`, `bgl_core.py`, and `bgl_data.py` to the baseline folder
on the server. IGB also uses the existing local `igb_dataloader.py`.
The standalone BGL loader follows the baseline construction but no longer
imports `benchmark_all_baselines.py`, avoiding version-dependent
`return_validation` interface errors.

```bash
# All four datasets sequentially, cuda:1, GraphSAGE, real features
python -u redd_bgl.py

# Just Paper and MAG (dataset-specific ratios and samplers are automatic)
python -u redd_bgl.py --dataset paper mag --epochs 1

# Normal end-to-end timing without per-stage synchronization
python -u redd_bgl.py --timing end-to-end

# One measured epoch after one warm-up, fixed sequence count
python -u redd_bgl.py --dataset reddit --epochs 1 --num-sequences 5 --cache-ratios 0.15

# Optional automatic sequence-count tuning; can be expensive
python -u redd_bgl.py --dataset reddit --num-sequences 0 --max-sequences 100
```

Filename `redd_bgl.py` is retained for compatibility, but default `--dataset`
is now `all`. Multiple explicit datasets are supported, e.g. `--dataset igb
reddit` or `--dataset paper mag`. Datasets run in separate sequential
subprocesses to release native graph/feature memory between datasets.

| Dataset | Cache ratios (original graph node count) | CPU fanout | Batch size | Training split |
| --- | --- | --- | --- | --- |
| IGB | 0.01, 0.02 | [5,10,15] | 2048 | 1% random subset of train mask, with original baseline self-loop preprocessing |
| Reddit | 0.15, 0.30 | [5,5,5] | 200 | Random 60% of nodes |
| Paper | 0.003, 0.008 | [5,10,15] | 2048 | Official train mask |
| MAG | 0.002, 0.005 | [15,15,15] | 2048 | Official training paper indices |

The BGL-specific ratio table above does not change the static baseline ratio
table in `benchmark_all_baselines.py`. `--cache-ratios` and `--batch-size`
explicitly override per-dataset defaults for every selected dataset. Data paths
default to `/lyz/data/IGB-dataset/`,
`/data_priv/yunzhen/ogbn-papers100M`, `/lyz/data`; use `--path`, `--paper-root`
or `--mag-root` to override. Paper uses its original OGB feature loading path;
MAG uses the original baseline float32 feature store. No mmap conversion or
synthetic feature mode is added.

All datasets use three-layer GraphSAGE with hidden size 256, dropout 0.5,
Adam lr=1e-3 and weight_decay=5e-4. Validation uses the Reddit random 20% split,
IGB's sampled validation mask, Paper's validation mask, or MAG's first half
of official valid indices, matching the baseline construction. It runs outside timing
and does not update the training FIFO. `--skip-validation` omits it.
`--gpu`, `--cache-ratios`, `--epochs`, `--output-file` are configurable.

## Proximity-aware seed ordering

1. At each epoch, independently build several BFS sequences with random roots.
2. Traverse the whole topology, including non-training intermediates, but emit
   only training seeds. Restart at random unvisited roots for disconnected
   components; isolated training nodes are included.
3. Circularly shift each sequence by an independent random offset.
4. Take one previously unselected seed per sequence in round-robin order until
   all training seeds are emitted. Each seed appears exactly once per epoch.
5. Split the resulting order into minibatches. DataLoader uses **shuffle=False**.
   `--check-order` verifies the actual minibatch output IDs match the order.

The paper describes multiple random-root BFS sequences, random shifts and
round-robin selection, but does not provide executable pseudocode for duplicate
handling, directed-edge orientation or label-TV aggregation. This adaptation
explicitly uses duplicate skipping, in-neighbor BFS by default (Reddit is
bidirectional), randomized disconnected-component roots and mean label-TV over
full minibatches. `--bfs-direction out` selects successor traversal.

Default `--num-sequences 5` follows the paper's example, not a guarantee that
five sequences satisfies a convergence criterion on Reddit. Every epoch logs
the empirical mean label-TV distance. Automatic mode increases the candidate
count up to the configurable pool limit (100 default), checking a label-TV
threshold. If no count passes, it reports `threshold_met=False` and uses the
best tested count; do not describe that as satisfying the convergence theorem.
`--tv-tolerance` permits an explicit experimental threshold. Selection and BFS
construction costs are outside the iteration breakdown and reported separately.

## Fixed GPU tensor and FIFO slot management

`FIFOFeatureCache.gpu_fix_cache` is allocated once as `[capacity, feature_dim]`.
It is not concatenated, shifted or reallocated on updates. External metadata:

- `cache_bucket[node_id]`: GPU slot index, or -1 for an uncached node.
- `slot_nodes[slot]`: node currently occupying the physical slot.
- `head`, `size`: FIFO ring head and number of occupied slots.

At feature retrieval, copy all hits into a separate current-batch input tensor,
read misses from the CPU feature store, and transfer them to GPU. Once the batch
input is assembled, enqueue its miss nodes in that order, evicting FIFO-head
entries as needed. This makes the previous batch's misses available to the next
batch. Hits do not refresh FIFO position. Reuse slots via indexed assignments
into the fixed GPU tensor and update the two ID maps.

If misses exceed capacity, logically all are enqueued but only the last C
survive. Physical writes are restricted to distinct slots to avoid duplicate
GPU indices and nondeterministic overwrites. Current-batch hit features are
gathered before eviction, so evicting a hit later in that batch cannot corrupt
its already prepared input. Reads and writes use the same current CUDA stream.
NeighborSampler input IDs are unique, as required by the cache interface.

Default initialization is `prefill_hotmap`, as requested: allocate the fixed
tensor, select hot nodes in a separate random presampling pass and populate
slots in chunks. All initial entries remain evictable. `--cache-init empty`
starts with no resident entries instead. The cache persists across epochs and
is reset, along with the model and optimizer, for each cache ratio.

## Results and timing

All results append to `bgl_all_results.txt`. Each epoch records dataset,
ratio, seed-sequence count, label-TV, cache hit ratio, loss and validation
accuracy. Measured-epoch summaries exclude warm-up.

Miss rates are also printed and written as fractions in [0,1]:
- `epoch_avg_miss_rate`: arithmetic mean of each batch's miss/request ratio,
  including every batch of that epoch (no first/last-batch exclusion).
- `epoch_weighted_miss_rate`: total missed node requests / total node requests.
- `summary_avg_miss_rate`: arithmetic mean over all measured batches across
  measured epochs, excluding warm-up epochs.
- `summary_weighted_miss_rate`: missed/requested nodes over those same batches.

The existing `cache_hit_rate` is the request-weighted hit rate; its complement
is the weighted miss rate, not necessarily the arithmetic batch average.

Breakdown records average sampling, block H2D, feature preparation **including
FIFO update**, forward/loss, backward/optimizer, and forward+backward times
(ms/iteration). Synchronized wall-clock measurement includes CPU map/queue work
and completed GPU operations. Complete E = block H2D + feature preparation.

End-to-end mode synchronizes only at epoch boundaries. Both modes log:
- `epoch_seconds_excluding_order`: training loop wall time;
- `order_seconds`: generation of fresh BFS sequences and mixing;
- `epoch_seconds_including_order`: their sum.

Label-TV computation and validation are diagnostics outside these times.
Do not compare a synchronized breakdown epoch against an asynchronous
end-to-end epoch as though they were the same throughput measurement.

## Small correctness checks

```bash
python -m unittest test_bgl_core.py -v
```

Tests use tiny CPU tensors and a mock adjacency. They check disconnected BFS
coverage, reproducibility, circular shifting, no duplicate/lost seeds, FIFO
hit behavior, wraparound, batch misses larger than capacity, feature correctness
and fixed cache storage. Target-server CUDA/DGL execution still needs checking.
