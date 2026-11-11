# Unified GNN Training and Cache Performance Benchmark

This repository provides a unified benchmark framework for evaluating **GPU feature caching efficiency** and **large-scale GNN training performance** on different graph datasets.

The framework contains two main components:

1. **Cache Hit Rate Evaluation**
2. **GNN Training Performance Evaluation**

---

The required environment dependencies for this project are provided in the env.yml file located in the same directory.

# 1. Cache Hit Rate Evaluation

## Overview

This module evaluates the effectiveness of different feature cache initialization strategies under various datasets.

The cache simulator models feature access patterns during GNN training and measures the cache miss rate under different cache capacities.

Supported cache initialization methods include:

- **Random Initialization**
- **Degree-based Initialization**
- **Hot-map-based Initialization**


The cache initialization strategy can be selected by modifying the corresponding cache initialization function in the code.

---

## Running Cache Evaluation

Cache hit rate experiments can be executed using scripts such as:

```bash
python storage_cpu_all_igb.py
```

The script evaluates different cache strategies on the target dataset and records the corresponding cache miss rates.

Example output:

```text
test_cache_radio: [...]
miss_random: [...]
miss_degree: [...]
miss_hot_map: [...]
miss_PreFC_d: [...]
miss_PreFC_h: [...]
```

The generated results can be used to compare the effectiveness of different cache initialization approaches.

---

## Cache Evaluation Workflow

The cache evaluation process contains the following steps:

1. Load the graph structure and training sampling sequence.
2. Generate node access patterns according to the training dataloader.
3. Initialize the cache using different strategies.
4. Simulate cache access during training iterations.
5. Calculate the cache miss rate under different cache sizes.

---

# 2. GNN Training Performance Evaluation

## Overview

This module evaluates the end-to-end performance of GNN training on different datasets.

The framework supports detailed profiling of the following components:

- Neighbor sampling
- Feature transfer between CPU and GPU
- Forward propagation
- Backward propagation
- Overall training iteration time

Supported GNN models include:

- GraphSAGE
- GCN
- GAT

---

## Performance Profiling Mode

The profiling behavior is controlled by the parameter:

```python
log_time
```

### Detailed Performance Breakdown

When:

```python
log_time=True
```

the framework records the execution time of each training stage:

```text
Sampling Time
Feature Transfer Time
Forward Propagation Time
Backward Propagation Time
Total Iteration Time
```

Example output:

```text
Avg SamplingOnly: xxx ms/iter
Avg Copy       : xxx ms/iter
Avg Forward    : xxx ms/iter
Avg Backward   : xxx ms/iter
Avg Total      : xxx ms/iter
```

---

### Overall Training Time

When:

```python
log_time=False
```

the framework only reports the overall epoch or iteration execution time.

This mode is suitable for measuring the end-to-end training throughput.

---

## Running Performance Evaluation

Example:

```bash
python mag_gpu_gate.py --mode puregpu --vis 5 --avg_input_size 250000
```

Common arguments:

| Argument | Description |
|----------|-------------|
| `--mode` | Training execution mode (`cpu`, `mixed`, `puregpu`) |
| `--vis` | Cache visibility window size |
| `--avg_input_size` | Average number of input nodes per iteration |
| `--log_time` | Enable detailed time profiling |

---

# File Naming Convention

The file names follow specific conventions to indicate the corresponding optimization techniques.

## `_gpu` Suffix

A filename with the suffix:

```text
_gpu
```

indicates that the sampling process is performed on the GPU.

Example:

```text
igb_gpu_gate.py
```

means that the training pipeline uses GPU-based sampling.

---

## `_gate` Suffix

A filename with the suffix:

```text
_gate
```

indicates that the training pipeline enables the hotness-based node partitioning mechanism.

The graph nodes are divided into hot and cold nodes according to their access frequency. This partitioning strategy is used to improve cache utilization and reduce feature transfer overhead.

Example:

```text
igb_gpu_gate.py
```

indicates:

- GPU-based sampling
- Hotness-aware node partitioning

---

# Supported Datasets

The framework supports experiments on multiple large-scale graph datasets:

- IGB260M
- MAG240M
- Reddit
- Paper100M

For each dataset, users can evaluate:

- Cache hit/miss performance
- Sampling overhead
- CPU-GPU feature transfer cost
- GNN computation efficiency
- Overall training performance

---

# Summary

This benchmark framework provides a unified environment for analyzing optimization techniques in large-scale GNN training.

The cache evaluation module focuses on:

- Cache initialization strategies
- Cache hit rate comparison
- Feature reuse efficiency

The performance evaluation module focuses on:

- Sampling efficiency
- Communication overhead
- Computation cost
- End-to-end training performance

Together, these components enable comprehensive evaluation of scalable GNN training systems.