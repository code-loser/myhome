> Random initialization is now the default (--cache-init random).
> IGB/Reddit draw uniformly without replacement from all original graph nodes.
> Paper/MAG draw only from the input-node IDs returned by ONE batch containing
> all training seeds, with dataset fanout and seed + 5. The exact same sampled
> pool defines both the capacity denominator and random initialization candidates.
> Each ratio takes a prefix of one random permutation (seed + 10); no duplicate
> IDs and no hot-map presampling in random mode. Cache replacement is unchanged.
> Results log cache_init=random and the pool size/source. Existing --cache-init
> hotmap and empty remain optional. Default run: python -u gids_cache_hit.py.

> Current GIDS admission policy (evaluation and training): count each miss
> node's future queue occurrences using torch.isin. Admit only positive-count
> candidates; truncate in current miss order to the number of zero-clock slots.
> Randomly choose that many zero-clock slots. No insufficient-slot assertion.
> Current-batch miss metrics and feature transfers still cover ALL misses.
> GPU writes select the exact admitted miss rows, which need not be a prefix.
> Incremental uint8 counters and queue order remain unchanged. Earlier strict
> assertion/bypass descriptions below are historical; rerun old measurements.

# GIDS original incremental-clock evaluation

```bash
python -u gids_cache_hit.py
python -u gids_cache_hit.py --dataset paper mag
```

The cache policy now follows igb_gids.py: torch cache_bucket, point, uint8 clock;
random zero-clock eviction; current hits decrement clock; newly inserted nodes
increment clock according to future queue membership. Queue order: prefill 8,
init_clock once, append next, pop current, fetch/update. No counter recomputation,
no bypass. Newly appended batches do not increment existing resident clocks,
matching the source. uint8 wraparound is also retained.

Two necessary initialization repairs remain: use actual configured capacity
instead of inconsistent hardcoded 1%/2%, and initialize point from prefilled IDs.
Empty slots never clear cache_bucket[-1]. Each capacity has its own torch RNG.
When misses exceed zero-clock slots, the original assertion fails with dataset,
capacity and batch information. Small-capacity runs may therefore fail.

Ratios/loaders/capacity bases are reused from unchanged bgl_cache_hit.py.
Paper/MAG use all-training-seed sampled input-node count; IGB/Reddit original N.
No feature loading, training or GPU allocation. Output appends to
 gids_cache_hit_results.txt; policy marker original_incremental_uint8_clock.
miss_rate is batch mean; weighted_miss_rate is total misses / total requests.
Previous bypass-policy results are not equivalent and need rerunning.

Only gids_cache_hit.py is rewritten. benchmark_all_baselines.py currently retains
its previous corrected metadata implementation; it is not changed by this task.
