"""Single-GPU BGL locality ordering and FIFO feature-cache components.

Reference: BGL, arXiv:2112.08541v1, sections 3.2.1--3.2.2, pp.5--6.
This reproduces the requested mechanisms, not BGL's distributed partitioning,
multi-GPU sharing, CPU remote-feature cache or resource-isolation system.
"""

import math
import time

import numpy as np
import torch


def round_robin_unique(sequences, shifts):
    """Alternate sequences one seed at a time, skipping already emitted IDs.

    All sequences must be permutations of the same training set. Circular
    shifts preserve each BFS sequence's consecutive order. Duplicate skipping
    is our explicit interpretation of the paper's otherwise underspecified
    mixing step, ensuring exactly one visit per training node per epoch.
    """
    if not sequences or len(sequences) != len(shifts):
        raise ValueError("Need one shift per nonempty sequence list")
    sequences = [np.asarray(s, dtype=np.int64) for s in sequences]
    n = len(sequences[0])
    if n == 0 or any(len(s) != n for s in sequences):
        raise ValueError("Sequences must have the same positive length")
    target = np.sort(sequences[0])
    if len(np.unique(target)) != n or any(not np.array_equal(np.sort(s), target)
                                        for s in sequences[1:]):
        raise ValueError("Each sequence must contain the same unique training IDs")
    seen = set()
    cursors = [0] * len(sequences)
    result = np.empty(n, dtype=np.int64)
    count = turn = 0
    while count < n:
        index = turn % len(sequences)
        while cursors[index] < n:
            node = int(sequences[index][(int(shifts[index]) + cursors[index]) % n])
            cursors[index] += 1
            if node not in seen:
                seen.add(node)
                result[count] = node
                count += 1
                break
        turn += 1
    return result


def label_tv_error(order, node_labels, train_nodes, batch_size):
    """Mean minibatch-vs-global label total-variation distance.

    Use full minibatches (or the only partial batch for very small datasets).
    The paper specifies label-frequency TV but not its minibatch aggregation;
    the mean-full-batch convention here is recorded in the run documentation.
    """
    global_labels = np.asarray(node_labels)[np.asarray(train_nodes)]
    if np.any(global_labels < 0):
        raise ValueError("All training seeds need nonnegative class labels")
    classes = int(global_labels.max()) + 1
    distribution = np.bincount(global_labels, minlength=classes) / len(global_labels)
    end = len(order) // batch_size * batch_size or len(order)
    errors = []
    for start in range(0, end, batch_size):
        labels = np.asarray(node_labels)[order[start:min(start + batch_size, end)]]
        local = np.bincount(labels, minlength=classes) / len(labels)
        errors.append(0.5 * np.abs(local - distribution).sum())
    return float(np.mean(errors))


class ProximityAwareOrdering:
    def __init__(self, graph, train_nodes, direction="in"):
        if graph.device.type != "cpu":
            raise ValueError("BFS ordering is constructed on the CPU graph")
        if direction not in ("in", "out"):
            raise ValueError("direction must be in or out")
        self.node_count = graph.num_nodes()
        self.train_nodes = train_nodes.long().cpu().numpy().copy()
        if not len(self.train_nodes) or len(np.unique(self.train_nodes)) != len(self.train_nodes):
            raise ValueError("Training seeds must be nonempty and unique")
        self.train_mask = np.zeros(self.node_count, dtype=np.bool_)
        self.train_mask[self.train_nodes] = True
        # CSC follows predecessors (same direction as neighbor sampling); CSR
        # follows successors. Reddit is bidirectional, so both are equivalent.
        indptr, indices, _ = graph.adj_tensors("csc" if direction == "in" else "csr")
        self.indptr = indptr.cpu().numpy()
        self.indices = indices.cpu().numpy()

    def bfs_sequence(self, rng):
        visited = np.zeros(self.node_count, dtype=np.bool_)
        queue = np.empty(self.node_count, dtype=np.int64)
        result = np.empty(len(self.train_nodes), dtype=np.int64)
        emitted = 0
        # Random root, and randomized restarts for disconnected components.
        for root in rng.permutation(self.node_count):
            if visited[root]:
                continue
            visited[root] = True
            queue[0] = root
            head, tail = 0, 1
            while head < tail:
                node = queue[head]
                head += 1
                if self.train_mask[node]:
                    result[emitted] = node
                    emitted += 1
                    if emitted == len(result):
                        return result
                neighbors = self.indices[self.indptr[node]:self.indptr[node + 1]]
                fresh = neighbors[~visited[neighbors]]
                if len(fresh):
                    # Deduplicate parallel edges before enqueuing/marking.
                    fresh = np.unique(fresh)
                    visited[fresh] = True
                    queue[tail:tail + len(fresh)] = fresh
                    tail += len(fresh)
        raise RuntimeError("BFS did not cover every training node")

    def epoch_order(self, count, seed):
        rng = np.random.RandomState(seed)
        sequences = [self.bfs_sequence(rng) for _ in range(count)]
        shifts = rng.randint(0, len(self.train_nodes), size=count)
        return torch.from_numpy(round_robin_unique(sequences, shifts))

    def select_sequence_count(self, labels, batch_size, seed, max_sequences=100,
                              tolerance=None):
        """Increase number of sequences until mean label-TV passes a threshold.

        Default threshold is sqrt(b*M)/n from paper p.6 with M=1. If none of
        the candidates passes, return the best tested count and explicitly
        report threshold_met=False; no convergence guarantee is claimed.
        """
        threshold = (math.sqrt(batch_size) / len(self.train_nodes)
                     if tolerance is None else tolerance)
        rng = np.random.RandomState(seed)
        sequences, shifts, trials = [], [], []
        best = None
        for count in range(1, max_sequences + 1):
            sequences.append(self.bfs_sequence(rng))
            shifts.append(int(rng.randint(len(self.train_nodes))))
            order = round_robin_unique(sequences, shifts)
            error = label_tv_error(order, labels, self.train_nodes, batch_size)
            trials.append((count, error))
            print("BGL sequence selection: count={} mean_label_TV={:.6f} threshold={:.8f}".format(
                count, error, threshold), flush=True)
            if best is None or error < best[1]:
                best = (count, error)
            if error <= threshold:
                return count, dict(threshold=threshold, threshold_met=True, trials=trials)
        return best[0], dict(threshold=threshold, threshold_met=False, trials=trials)


class FIFOFeatureCache:
    """Fixed feature tensor + CPU node/slot map and FIFO ring metadata.

    Cache hits NEVER refresh position. After assembling the current batch,
    append its unique miss nodes; they become cached for subsequent batches.
    This is FIFO (the requested approximate-LRU), not true LRU. Inputs from
    DGL NeighborSampler contain unique node IDs; enqueue requires that too.
    All CUDA accesses use the caller's current stream, with no worker races.
    """
    def __init__(self, features, capacity, device):
        self.features = features
        self.capacity = int(capacity)
        self.device = torch.device(device)
        if self.capacity < 0 or self.capacity > features.shape[0]:
            raise ValueError("Invalid cache capacity")
        self.gpu_fix_cache = torch.empty((self.capacity, features.shape[1]),
                                         dtype=torch.float32, device=self.device)
        self.cache_bucket = torch.full((features.shape[0],), -1, dtype=torch.long)
        self.slot_nodes = torch.full((self.capacity,), -1, dtype=torch.long)
        self.head = self.size = 0

    def prefill_hotmap(self, hotmap, chunk_size=65536):
        if self.size:
            raise ValueError("Prefill only an empty cache")
        if not self.capacity:
            return
        nodes = torch.topk(hotmap, self.capacity, largest=True).indices
        for start in range(0, self.capacity, chunk_size):
            selected = nodes[start:start + chunk_size]
            self.enqueue(selected, self.features[selected].float().to(self.device))

    def fifo_node_ids(self):
        if not self.size:
            return torch.empty(0, dtype=torch.long)
        slots = (torch.arange(self.size) + self.head) % self.capacity
        return self.slot_nodes[slots].clone()

    def enqueue(self, nodes, gpu_values):
        """Append unique currently uncached IDs, preserving their input order."""
        nodes = nodes.long().cpu()
        count = nodes.numel()
        if not count or not self.capacity:
            return
        if count >= self.capacity:
            # Equivalent to enqueuing ALL misses one by one: only the last C
            # can survive. Avoid duplicate GPU destination indices in a write.
            occupied = self.slot_nodes[self.slot_nodes >= 0]
            self.cache_bucket[occupied] = -1
            nodes = nodes[-self.capacity:]
            gpu_values = gpu_values[-self.capacity:]
            slots = torch.arange(self.capacity)
            self.head, self.size = 0, self.capacity
        else:
            evictions = max(0, self.size + count - self.capacity)
            if evictions:
                old_slots = (torch.arange(evictions) + self.head) % self.capacity
                self.cache_bucket[self.slot_nodes[old_slots]] = -1
                self.head = (self.head + evictions) % self.capacity
                self.size -= evictions
            tail = (self.head + self.size) % self.capacity
            slots = (torch.arange(count) + tail) % self.capacity
            self.size += count
        self.gpu_fix_cache[slots.to(self.device)] = gpu_values
        self.slot_nodes[slots] = nodes
        self.cache_bucket[nodes] = slots

    def fetch_and_update(self, input_nodes, output_nodes, labels):
        nodes = input_nodes.long().cpu()
        hit_mask = self.cache_bucket[nodes] >= 0
        hits = torch.nonzero(hit_mask, as_tuple=True)[0]
        misses = torch.nonzero(~hit_mask, as_tuple=True)[0]
        x = torch.empty((nodes.numel(), self.features.shape[1]),
                        dtype=torch.float32, device=self.device)
        # Gather all hits BEFORE reusing any cache slots. x owns its storage.
        if hits.numel():
            x[hits.to(self.device)] = self.gpu_fix_cache[
                self.cache_bucket[nodes[hits]].to(self.device)]
        miss_nodes = nodes[misses]
        miss_values = self.features[miss_nodes].float().to(self.device)
        x[misses.to(self.device)] = miss_values
        y = labels[output_nodes.long().cpu()].long().to(self.device)
        update_start = time.perf_counter()
        self.enqueue(miss_nodes, miss_values)
        # Host enqueue duration is diagnostic only (not completed CUDA time).
        return x, y, hits.numel(), misses.numel(), time.perf_counter() - update_start
