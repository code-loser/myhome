# Unified GNN Training and Performance Analysis Framework (Unified GNN Benchmark)

Supports accurate runtime profiling of sampling, feature transfer, GPU caching, and forward/backward propagation for various GNN models (GraphSAGE, GCN, GAT) across different datasets (IGB260M, MAG240M, Reddit, Paper100M).
For model or training configurations with the suffix _gpu, the corresponding sampling process is performed on the GPU.

---


## Quick Start

Run the default configuration (uand training SAGE, GCN, and GAT models sequentially .The desired cache initialization method can be configured by adjusting the corresponding cache initialization function in the code.):


```bash
python redd.py --mode puregpu