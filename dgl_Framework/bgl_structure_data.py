"""Feature-free raw graph loaders for BGL cache-policy simulation.

Never call full DGL/OGB dataset loaders or access feature NPZ members.
Topology and labels still occupy RAM. NPZ compressed members are loaded
normally, not converted to NPY or claimed to support mmap.
"""

from pathlib import Path

import dgl
import numpy as np
import torch


def labels_tensor(array):
    flat = np.asarray(array).reshape(-1)
    result = torch.empty(flat.size, dtype=torch.long)
    for start in range(0, flat.size, 1_000_000):
        values = np.array(flat[start:start + 1_000_000], copy=True)
        if np.issubdtype(values.dtype, np.floating):
            values[~np.isfinite(values)] = -1
        result[start:start + values.size] = torch.from_numpy(values).long()
    return result


def find_paper_dir(root):
    root = Path(root)
    for base in (root, root / "ogbn_papers100M", root / "ogbn-papers100M"):
        if (base / "raw" / "data.npz").is_file():
            return base
    raise FileNotFoundError("Cannot find Papers100M raw/data.npz under {}".format(root))


def find_reddit_dir(root):
    if root:
        candidates = [Path(root), Path(root) / "reddit"]
    else:
        from dgl.data.utils import get_download_dir
        base = Path(get_download_dir())
        candidates = [base / "reddit", base] + sorted(base.glob("reddit*"))
    for base in candidates:
        if (base / "reddit_graph.npz").is_file() and (base / "reddit_data.npz").is_file():
            return base
    raise FileNotFoundError(
        "Feature-free Reddit requires raw reddit_graph.npz and reddit_data.npz; "
        "provide their directory with --reddit-root. Processed DGL graph files "
        "are deliberately not loaded because they also contain node features.")


def load_structure(name, args):
    if name == "igb":
        # IGB260M's properties load only the requested array. Do not access
        # paper_feat or instantiate the full feature-loading DGL wrapper.
        from igb_dataloader import IGB260M
        raw = IGB260M(args.path, args.dataset_size, args.in_memory,
                      args.num_classes, 0)
        edges = torch.from_numpy(raw.paper_edge)
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=raw.num_nodes())
        graph = dgl.add_self_loop(graph)  # consistent with BGL training loader
        labels = labels_tensor(raw.paper_label)
        permutation = torch.randperm(graph.num_nodes())
        original_train = permutation[:int(graph.num_nodes() * 0.6)]
        count = max(1, int(original_train.numel() * args.sample_frac))
        train = original_train[torch.randperm(original_train.numel())[:count]].clone()
    elif name == "reddit":
        import scipy.sparse
        base = find_reddit_dir(args.reddit_root)
        adjacency = scipy.sparse.load_npz(str(base / "reddit_graph.npz"))
        graph = dgl.from_scipy(adjacency)
        with np.load(str(base / "reddit_data.npz"), allow_pickle=False) as archive:
            labels = labels_tensor(archive["label"])
            # archive['feature'] is never accessed.
        train = torch.randperm(graph.num_nodes())[:int(graph.num_nodes() * 0.6)].clone()
    elif name == "paper":
        base = find_paper_dir(args.paper_root)
        with np.load(str(base / "raw" / "data.npz"), allow_pickle=False) as archive:
            node_count = int(archive["num_nodes_list"].reshape(-1)[0])
            edges = archive["edge_index"]
            # archive['node_feat'] and archive['node_year'] are never accessed.
        graph = dgl.graph((torch.from_numpy(edges[0]), torch.from_numpy(edges[1])),
                          num_nodes=node_count)
        with np.load(str(base / "raw" / "node-label.npz"), allow_pickle=False) as archive:
            labels = labels_tensor(archive["node_label"])
        split = base / "split" / "time" / "train.csv.gz"
        train = torch.from_numpy(np.loadtxt(str(split), dtype=np.int64,
                                           delimiter=",").reshape(-1)).long()
    elif name == "mag":
        from ogb.lsc import MAG240MDataset
        raw = MAG240MDataset(root=args.mag_root)
        edges = raw.edge_index("paper", "paper")
        graph = dgl.graph((torch.from_numpy(edges[0]), torch.from_numpy(edges[1])),
                          num_nodes=raw.num_papers)
        labels = labels_tensor(raw.paper_label)
        train = torch.from_numpy(raw.get_idx_split("train")).long()
        # raw.paper_feat is never accessed.
    else:
        raise ValueError(name)
    if labels.numel() != graph.num_nodes() or not train.numel():
        raise ValueError("Invalid label or training-node count")
    if (labels[train] < 0).any():
        raise ValueError("Training set contains unlabeled nodes")
    return graph, labels, train
