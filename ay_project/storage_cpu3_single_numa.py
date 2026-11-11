"""Single-NUMA cache worker; paired with igb_gpu_single_numa.py."""
import torch
import dgl
from collections import deque
from typing import Dict, Optional, List
import sys
import base64
import os
sys.path.append('/ay_project/cuda_ipc')
import cuda_ipc_get
import cuda_ipc_open
import numpy as np
import cupy as cp
import time
from igb_dataloader import IGB260MDGLDataset_part
from ogb.lsc import MAG240MDataset
import os.path as osp
from torch_geometric.datasets import Yelp
import torch_geometric.transforms as T
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.data import RedditDataset
import psutil
from concurrent.futures import ThreadPoolExecutor


class BlockFeatureCacheServer:
    def __init__(self, node_num , ndata ,args ,gpuid: Optional[int], feature_field: str = 'feat', label_field: str = 'label', window_size: int = 3):
        self.gpuid = gpuid
        self.dtype = torch.float32 if args.dtype == "float32" else torch.float16
        self.id_type = torch.long
        self.feature_field = feature_field
        self.label_field = label_field
        self.feat_dim = int(ndata.shape[1])
        self.node_num = node_num
        self.full_cached = False
        # Ratios use the original graph's node count, not training seeds or
        # the sampling subgraph's active nodes. Entry points pass g.num_nodes().
        cache_ratios = {"re": 0.15, "IGB260M": 0.01,
                        "paper": 0.008, "mag": 0.005}
        if args.feat_name not in cache_ratios:
            raise ValueError("No single-NUMA cache ratio configured for: "
                             + args.feat_name)
        self.cache_ratio = cache_ratios[args.feat_name]
        self.max_cache_size = int(self.cache_ratio * node_num)
        if self.max_cache_size < 1:
            raise ValueError("Cache capacity must be at least one node")
        print("[cache] dataset={} original_nodes={} cache_ratio={} cache_nodes={}".format(
            args.feat_name, node_num, self.cache_ratio, self.max_cache_size), flush=True)
        self.device = torch.device('cuda', self.gpuid) if self.gpuid is not None else torch.device('cpu')
        self.prefill_num = int(self.max_cache_size * 0.5)
        self.Visibility = args.vis
        self.limits = 0
        self.Fixed = int(self.max_cache_size * 0.40)
        self.gpu_flag = torch.zeros(self.node_num, dtype=torch.bool)
        self.cached_num = 0
        self.capability = self.node_num
        self.gpu_fix_cache = None
        self.use_uva = args.use_uva
        self.cache_bucket = None
        self.feat_name = args.feat_name
        self.task_queue = torch.empty(0, dtype=self.id_type)
        self.point = torch.empty(0, dtype=self.id_type)
        self.point_position = torch.empty(0, dtype=self.id_type)
        self.offset_list = []
        self.cache_capacity = 0
        self.find_time = 0.0
        self.sort_time = 0.0
        self.exchange_time = 0.0
        self.miss_rate = 0.0
        self.ndata = ndata
        self.prefill = torch.empty(0, dtype=self.id_type)
        self.avg_input_size = args.avg_input_size
        self.feature = torch.full((self.avg_input_size, self.feat_dim), fill_value=0 ,dtype=self.dtype, device=self.device)
        self.pinned_buffer = torch.empty((self.avg_input_size , self.feat_dim), dtype=self.dtype).pin_memory()
        self.feature_need_push = None
        
        

    def prefill_by_degree(self):
        self.gpu_fix_cache = torch.empty(
            (self.max_cache_size, self.feat_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=self.id_type,
            device=self.device if self.use_uva else "cpu"
        )
        degrees = self.prefill[self.node_num:2*self.node_num]
        topk_vals, topk_idx = torch.topk(degrees, k=self.max_cache_size, largest=True)
        slots = torch.arange(0, self.max_cache_size, dtype=self.id_type , device=self.cache_bucket.device)
        feats = self.ndata[topk_idx].to(self.device, non_blocking=True)
        topk_idx = topk_idx.to(self.cache_bucket.device)
        self.point = topk_idx[self.Fixed:]
        self.gpu_fix_cache[slots.to(self.device)] = feats
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.num_iter = 0.0
        self.time = 0.0
        self.wait_time = 0.0
        self.evict_time = 0.0
        self.zuz = 0.0

    def prefill_by_hotmap(self):
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=self.id_type,
            device=self.device if self.use_uva else "cpu"
        )
        hot_map = self.prefill[:self.node_num]
        topk_vals, topk_idx = torch.topk(hot_map, k=self.max_cache_size, largest=True)
        self.point = topk_idx[self.Fixed:]
        self.gpu_fix_cache = self.ndata[topk_idx].to(self.device, non_blocking=True)
        slots = torch.arange(0, self.max_cache_size, dtype=self.id_type, device=self.cache_bucket.device)
        topk_idx = topk_idx.to(self.cache_bucket.device)
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.num_iter = 0
        self.time = 0.0
        self.wait_time = 0.0
        self.evict_time = 0.0
        self.zuz = 0.0
        
    
    def _evict_slots2(self, nids_new,feats, step):
        if nids_new.numel() == 0:
            return 
        self.cache_bucket[nids_new] = torch.arange(self.max_cache_size, self.max_cache_size + nids_new.numel(), dtype=self.id_type ,device = self.cache_bucket.device)
        result = self.find_first_indices1(nids_new, step)
        self.cache_bucket[nids_new] = -1
        
        keep_idx = torch.where(result[self.max_cache_size:] != self.Visibility)[0]
        drop_idx = torch.where(result[self.Fixed:self.max_cache_size] == self.Visibility)[0]
        
        if drop_idx.numel() == 0 or keep_idx.numel() == 0:
            return
        exchange_num = min(drop_idx.numel(), keep_idx.numel())
        if keep_idx.numel() <= drop_idx.numel():
            idx = torch.randperm(keep_idx.numel())[:exchange_num]
            evict_idx = drop_idx[idx]
            insert_idx = keep_idx
        else:
            evict_idx = drop_idx
            min_vals,insert_idx = torch.topk(result[self.max_cache_size:], k=exchange_num, largest=False , sorted=False)
            insert_idx = insert_idx

        candidates = nids_new[insert_idx]
        evict_nids = self.point[evict_idx]
        self.point[evict_idx] = candidates
        value = self.cache_bucket[evict_nids]
        self.cache_bucket[evict_nids] = -1
        self.cache_bucket[candidates] = value
        self.gpu_fix_cache[value] = feats[insert_idx]
    
    def find_first_indices1(self, a: torch.Tensor, step):
        result = torch.full((self.max_cache_size + a.size(0),), fill_value=self.Visibility, dtype=self.id_type, device=self.cache_bucket.device)
        kk = min(self.Visibility - 1 , self.limits - step -1)
        while kk >= 0:
            idx = (step + kk) % (self.Visibility + 1)
            b = self.task_queue[idx*self.avg_input_size:idx *self.avg_input_size+self.offset_list[idx]]
            bucket = self.cache_bucket[b]
            mask = bucket >= self.Fixed
            result[bucket[mask]] = kk
            kk -= 1
        return result


    def worker(self, handle_dict, task_queue, signal ,way, ii):
        for i in range(self.epoch_count):
            self.miss_rate = 0.0
            while task_queue.empty():
                pass
            item = task_queue.get()
            self.limits= item
            step = 0
            evict_idx = None
            insert_idx = None
            while step < self.limits:
                start_time = time.time()
                idx = step % (self.Visibility+1)
                input_nodes = self.task_queue[idx*self.avg_input_size:idx *self.avg_input_size+self.offset_list[idx]]
                if input_nodes.numel() > self.avg_input_size:
                    raise RuntimeError("Increase --avg_input_size to fit sampled input nodes")
                cache_mask = self.cache_bucket[input_nodes] != -1
                nids_cpu = input_nodes[~cache_mask]
                nids_cuda = input_nodes[cache_mask]
                self.miss_rate += nids_cpu.numel()/input_nodes.numel()
                on_cache_idx = self.cache_bucket[nids_cuda]
                start_trans = time.time()
                self.pinned_buffer[:nids_cpu.numel()].copy_(self.ndata[nids_cpu], non_blocking=True)
                end_trans = time.time()
                self.num_iter += end_trans - start_trans
                feat_need_push = self.pinned_buffer[:nids_cpu.numel()].to(self.device, non_blocking=True)
                self.feature[0:input_nodes.numel()][~cache_mask] = feat_need_push
                start_wait = time.time()
                self.feature[0:input_nodes.numel()][cache_mask] = self.gpu_fix_cache[on_cache_idx]
                handle_dict.put(input_nodes.numel() ,block=False)
                while signal.empty() and step < self.limits - self.Visibility:
                    pass
                end_wait = time.time()
                self.wait_time += start_trans - start_time
                if step + self.Visibility < self.limits:
                   iitm = signal.get()
                if way == 1:
                    self._evict_slots2(nids_cpu,feat_need_push ,step + 1)
                    self.evict_time += time.time() - end_wait
                if step == 0:
                    self.miss_rate -= nids_cpu.numel()/input_nodes.numel()
                step += 1
                self.time += time.time() - start_time
            self.miss_rate -= nids_cpu.numel()/input_nodes.numel()
            txt_path = f"time_for_{str(self.feat_name)}.log"
            if i%3 == 0:
                with open(txt_path, "a") as f:
                    f.write(f"{i}\n")
                    f.write(f"miss_rate: {self.miss_rate/(self.limits-2)}\n")
                    f.write(f"avg time: {self.time/self.limits}\n")
                    f.write(f"avg wait_time: {self.wait_time/(self.limits)}\n")
                    f.write(f"avg evict_time: {self.evict_time/(self.limits - self.Visibility)}\n")
                    f.write(f"avg trans_time: {self.num_iter/self.limits}\n")
            self.time = 0.0
            self.wait_time = 0.0
            self.evict_time = 0.0
            self.num_iter = 0.0
            self.zuz = 0.0

def load_feats(args):
    if args.feat_name == "yelp":
        dataset_name = 'Yelp'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '/data/data', dataset_name) 
        dataset = Yelp(path, transform=T.NormalizeFeatures())
        data = dataset[0]
        paper_feat = data.x
        return paper_feat
    elif args.feat_name == "mag":
        dataset = MAG240MDataset(root=getattr(args, "mag_root", "/lyz/data"))
        paper_feat = torch.from_numpy(dataset.paper_feat)
        return paper_feat
    elif args.feat_name == "IGB260M":
        dataset = IGB260MDGLDataset_part(args , only_feat=True)
        paper_feat = dataset[0]
        return paper_feat
    elif args.feat_name == "paper":
        node_num = 111059956
        # Diagnostic: match the original Papers100M float32 [N,128] features,
        # without loading a dataset graph or reading/converting feature files.
        if node_num is None or node_num < 1:
            raise ValueError("Paper zero features require the original graph node count")
        dtype = torch.float32
        size_gib = node_num * 128 * 4 / 1024**3
        print("[cache DIAGNOSTIC] creating CPU zeros shape=({},128) dtype={} "
              "size={:.2f} GiB".format(node_num, dtype, size_gib), flush=True)
        paper_feat = torch.zeros((node_num, 128), dtype=dtype, device="cpu")
        print("[cache DIAGNOSTIC] zero-feature allocation completed", flush=True)
        return paper_feat 
    elif args.feat_name == "re":
        data = RedditDataset()
        g = data[0]
        paper_feat = g.ndata["feat"]
        return paper_feat
def cache_worker_main(args ,node_num, gpuid, handle_dict, task_queue, signal):
    configure_single_numa(args, "cache")
    torch.cuda.set_device(args.gpu)
    prefill = ["hotmap", "degree"]
    paper_feat = load_feats(args)
    print("[cache] feature shape={}".format(paper_feat.shape), flush=True)
    cache_server = BlockFeatureCacheServer(node_num , paper_feat,args,gpuid )
    handle1 = cuda_ipc_get.get_cuda_ipc_handle(cache_server.feature)
    handle_b1 = base64.b64encode(handle1)
    while task_queue.empty():
            pass
    item = task_queue.get()
    cache_server.prefill = item
    handle_dict.put((handle_b1 , cache_server.feature.shape))
    while task_queue.empty():
        pass
    item = task_queue.get()
    if args.use_uva:
        handle_1 , shape1 , cache_server.offset_list = item
        handle_nodes = base64.b64decode(handle_1)
        cache_server.task_queue = cuda_ipc_open.open_cuda_ipc_handle(handle_nodes, list(shape1), args.gpu , "int64")
    else:
        cache_server.task_queue , cache_server.offset_list = item
    cache_server.epoch_count = args.epochs + 1
    range_list = [0] if args.cache_init == "hotmap" else [1]
    for ii in range_list:
          if ii == 0:
              cache_server.prefill_by_hotmap()
          elif ii == 1:
              cache_server.prefill_by_degree()
          cache_server.worker(handle_dict, task_queue, signal , 1 ,prefill[ii])


# Single-NUMA launch/configuration helpers (shared by the two new files).
def configure_single_numa(args, role):
    import subprocess
    from pathlib import Path

    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Single-NUMA experiments require Linux")
    policy = subprocess.run(["numactl", "--show"], capture_output=True,
                            text=True, check=True).stdout
    fields = dict(line.split(":", 1) for line in policy.splitlines() if ":" in line)
    if (fields.get("policy", "").strip() != "bind" or
            fields.get("membind", "").split() != [str(args.numa_node)]):
        raise RuntimeError("Launch with numactl --cpunodebind=N --membind=N; "
                           "N must equal --numa-node. Actual policy: " + policy)
    node_path = Path("/sys/devices/system/node/node{}".format(args.numa_node))
    node_cpus = set()
    for part in (node_path / "cpulist").read_text().strip().split(","):
        bounds = [int(x) for x in part.split("-")]
        node_cpus.update(range(bounds[0], bounds[-1] + 1))
    if role == "main":
        # Save the launch pool before narrowing parent affinity. The child may
        # widen its inherited affinity to its disjoint subset of the SAME node.
        allowed = node_cpus & set(os.sched_getaffinity(0))
        physical = {}
        for cpu in sorted(allowed):
            topology = Path("/sys/devices/system/cpu/cpu{}/topology".format(cpu))
            key = ((topology / "physical_package_id").read_text().strip(),
                   (topology / "core_id").read_text().strip())
            physical.setdefault(key, cpu)
        args.numa_cpu_pool = sorted(physical.values())
    pool = args.numa_cpu_pool
    required = args.main_threads + args.cache_threads
    if required > len(pool):
        raise RuntimeError("Requested {} physical cores, but launch pool has {}".format(required, len(pool)))
    cpus = (pool[:args.main_threads] if role == "main" else
            pool[args.main_threads:required])
    if not set(cpus) <= node_cpus:
        raise RuntimeError("CPU assignment is outside the selected NUMA node")
    # Pin existing library threads too; newly created threads inherit affinity.
    for thread in psutil.Process().threads():
        try:
            os.sched_setaffinity(thread.id, cpus)
        except ProcessLookupError:
            pass
    count = args.main_threads if role == "main" else args.cache_threads
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(count)
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)
    print("[{}] pid={} numa_node={} cpus={} torch_threads={} memory_policy=bind".format(
        role, os.getpid(), args.numa_node, cpus, torch.get_num_threads()), flush=True)

