> Capacity denominator update: Paper/MAG use `int(graph_nodes.numel() * ratio)`.
> `graph_nodes` comes from ONE CPU DataLoader batch containing all training seeds,
> with shuffle=True and NeighborSampler([5,10,15]) for Paper or [15,15,15] for MAG.
> All ratios share that single sampled denominator (seed + 5), logged as
> cache_capacity_base_nodes. IGB/Reddit still use the original graph node count.
> Earlier result blocks used the old denominator; rerun Paper/MAG for new results.

> Current configuration: `bgl_cache_hit.py` fixes BFS sequence count to **5**.
> Cache ratios follow the first worksheet of `PreGNNFlow-DATA-revised.xlsx`
> (Reddit row 4, IGB row 12, Papers row 20, MAG row 28), totaling 51 configurations.
> `--cache-ratios` is no longer accepted; old sequence-sweep commands below are historical.
>
> Run all datasets: `python -u bgl_cache_hit.py`
>
> Run selected datasets: `python -u bgl_cache_hit.py --dataset paper mag`
>
> Output: `bgl_cache_hit_results.txt` beside the script (append mode).
> Each row includes dataset, sequences=5, cache_ratio, cache_nodes, miss_rate,
> and weighted_miss_rate. `miss_rate` is the mean of per-batch miss fractions;
> weighted_miss_rate is total misses / total requests. All batches count by default.
> No feature loading, GPU allocation or training. Excel is not needed at runtime.

# Feature-free BGL sequence-count/cache-capacity experiment

```bash
# All four graphs, CPU only, default nine sequence counts
python -u bgl_cache_hit.py

# Pick graphs and sequence counts
python -u bgl_cache_hit.py --dataset paper mag --sequences 1 2 4 5 8

# Example: use the smaller Paper ratios from the BGL timing experiment
python -u bgl_cache_hit.py --dataset paper --cache-ratios 0.003 0.008
```

Deploy `bgl_cache_hit.py`, `bgl_structure_data.py`, `bgl_core.py`, and the
existing local `igb_dataloader.py`. No changes to the training entry or static
baseline scripts are required. There is no GPU requirement or model training.

| Dataset | Default cache ratios | Fanout | Batch size |
| --- | --- | --- | --- |
| IGB | 0.01, 0.02 | [5,10,15] | 2048 |
| Reddit | 0.15, 0.30 | [5,5,5] | 200 |
| Paper | 0.05, 0.10 | [5,10,15] | 2048 |
| MAG | 0.03, 0.06 | [15,15,15] | 2048 |

Ratios default to ay_project/cache_hit_vs_k.py, as selected for this experiment,
not redd_bgl.py's smaller Paper/MAG defaults. Capacity always uses the original
graph's node count. `--cache-ratios X Y` explicitly overrides both values for
the selected graphs. Sequence counts default to 1,2,4,5,6,8,10,12,14; override
with `--sequences`. The original PO seed-ordering and batch FIFO rules are kept.

## What is loaded

- IGB: only the raw loader's edge and label properties; no `paper_feat` access.
  Use the baseline's self loops, random 60% training split and its 1% subset.
- Reddit: `reddit_graph.npz` sparse adjacency and the `label` member of
  `reddit_data.npz`. The `feature` member is never read. Locate the raw files
  automatically in the DGL download directory, or pass `--reddit-root DIR`.
  The graph uses raw node IDs; DGL's processed cache may have a different node
  permutation. Thus this is structurally equivalent but its seeded random
  training split need not be identical to a reordered `RedditDataset()` split.
- Paper: raw `data.npz` members `edge_index` and `num_nodes_list`, raw labels,
  and official time-based training IDs. `node_feat`/`node_year` are not read.
- MAG: paper-paper topology, labels and official training IDs only.

Default paths are the same as the previous scripts: IGB
`/lyz/data/IGB-dataset/`, Paper `/data_priv/yunzhen/ogbn-papers100M`, MAG
`/lyz/data`. No NPY conversion files or full-feature dataset loaders are used.
The raw archives must already exist. Topology, labels, BFS workspaces and FIFO
metadata still occupy CPU RAM; feature-free does not mean memory-free.

## Fairness and metrics

One separate random presampling pass constructs a hot-map. Each sequence-count
configuration initializes its two caches from the same top-hotness selection.
`--cache-init empty` instead starts cold. No permanently fixed cache partition
is used: this is BGL-style FIFO, not the FNSL replacement in cache_hit_vs_k.py.

Build max(sequence counts) independent BFS sequences once. Each tested count
uses the matching prefix and the same prefix of circular shifts; the BFS/mixing
algorithm is the same as training, but paired RNG streams are intentionally
used for this sensitivity experiment. Every training seed appears exactly once.
The two capacities process the SAME sampled request batch before each updates
its own FIFO. No feature array, feature cache tensor or access trace is stored.

All results append to `bgl_cache_hit_results.txt`, with dataset sections and
rows labelled by sequence count, ratio and cache-node capacity:

- `hit_rate`: arithmetic mean of per-batch hit rates.
- `miss_rate`: 1 - hit_rate.
- `weighted_hit_rate`: total hit node requests / total node requests.
- `weighted_miss_rate`: 1 - weighted_hit_rate.
- `mean_label_TV`: label-distribution diagnostic for that seed order.

By default include all batches, including first/last. This differs from the
endpoint-excluded macro metric in cache_hit_vs_k.py; denominators are explicit.
`--skip-batches N` updates cache for the first N batches but excludes them from
statistics. Each count starts fresh; no state carries across counts. These are
one-epoch policy statistics, not GPU throughput measurements.

Datasets run sequentially in independent subprocesses. A failure is logged;
remaining graphs can still run and the driver finishes with nonzero status.
