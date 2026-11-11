"""Self-contained BGL dataset loader; independent of baseline entry versions.

Matches the four non-GPU baseline loaders and returns both training and
validation seeds. Uses real features, never synthetic diagnostic features.
"""
import random
import dgl
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def load_bgl_dataset(name, args):
    """Keep loading and train splits consistent with non-GPU baseline files."""
    seed_everything(args.seed)
    if name == "igb":
        from igb_dataloader import IGB260MDGLDataset

        dataset = IGB260MDGLDataset(args)
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        graph = dgl.add_self_loop(graph).to("cpu")  # igb.py preprocessing
        full_train = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
        count = max(1, int(full_train.numel() * args.sample_frac))
        train_idx = full_train[torch.randperm(full_train.numel())[:count]]
        classes = args.num_classes
    elif name == "reddit":
        from dgl.data import RedditDataset

        graph = RedditDataset()[0]
        order = torch.randperm(graph.num_nodes())
        train_idx = order[:int(graph.num_nodes() * 0.6)].clone()
        classes = int(graph.ndata["label"].max().item()) + 1
    elif name == "paper":
        from dgl.data import AsNodePredDataset
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root=args.paper_root),
            save_dir=args.paper_root, verbose=True)
        graph = dataset[0]
        if isinstance(graph, list):
            graph = graph[0]
        train_idx = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
        classes = 172
    elif name == "mag":
        from ogb.lsc import MAG240MDataset

        dataset = MAG240MDataset(root=args.mag_root)
        edges = dataset.edge_index("paper", "paper")
        graph = dgl.graph((torch.from_numpy(edges[0]), torch.from_numpy(edges[1])),
                          num_nodes=dataset.num_papers)
        # Match mag.py's float32 feature store, not synthetic or GPU-only data.
        graph.ndata["feat"] = torch.from_numpy(dataset.paper_feat).float()
        graph.ndata["label"] = torch.from_numpy(dataset.paper_label).long()
        train_idx = torch.from_numpy(dataset.get_idx_split("train")).long()
        classes = 153
    else:
        raise ValueError(name)
    if name == "reddit":
        val_idx = order[int(graph.num_nodes() * 0.6):int(graph.num_nodes() * 0.8)].clone()
    elif name == "mag":
        valid = torch.from_numpy(dataset.get_idx_split("valid")).long()
        val_idx = valid[:valid.numel() // 2].clone()
    else:
        val_idx = torch.nonzero(graph.ndata["val_mask"], as_tuple=True)[0]
        if name == "igb":
            count = max(1, int(val_idx.numel() * args.sample_frac))
            val_idx = val_idx[torch.randperm(val_idx.numel())[:count]]
    features = graph.ndata["feat"]
    labels = graph.ndata["label"].reshape(-1).long()
    # Preserve the feature store separately so node_subgraph cannot duplicate
    # features or labels during degree initialization. Sampling semantics stay
    # unchanged because only the graph topology is used by NeighborSampler.
    for key in list(graph.ndata.keys()):
        del graph.ndata[key]
    for key in list(graph.edata.keys()):
        del graph.edata[key]
    if features.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("Baseline feature store and labels must be on CPU")
    if train_idx.numel() == 0:
        raise ValueError("Empty training split")
    return graph, features, labels, train_idx, classes, val_idx

