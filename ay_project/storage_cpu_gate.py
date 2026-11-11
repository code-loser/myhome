from typing import Dict, Optional, List
import sys
import base64
import os
import torch
import dgl
sys.path.append('/ay_project/cuda_ipc')
import slabhash
import slabset
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
        self.max_cache_size = int(0.003 * node_num)
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
        self.pinned_buffer = torch.empty((self.avg_input_size , self.feat_dim), dtype=self.dtype).pin_memory()
        self.feature = torch.full((self.avg_input_size, self.feat_dim), fill_value=0 ,dtype=self.dtype, device=self.device)
        self.gate_mask = torch.empty(0, dtype=self.id_type)
        self.real_len = torch.empty(0, dtype=self.id_type)
        self.gate_rate = 0.0
        self.evit_num = 0
        
        

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
            device="cuda:0" if self.use_uva else "cpu"
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
        slots = torch.arange(0, self.max_cache_size, dtype=self.id_type, device=self.cache_bucket.device)
        self.gpu_fix_cache = self.ndata[topk_idx].to(self.device, non_blocking=True)
        topk_idx = topk_idx.to(self.cache_bucket.device)
        self.point = topk_idx[self.Fixed:]
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.num_iter = 0
        self.time = 0.0
        self.wait_time = 0.0
        self.evict_time = 0.0
        self.zuz = 0.0

    
    def _evict_slots2(self, nids_new, feats, step):
        if nids_new.numel() == 0:
            return 
        self.cache_bucket[nids_new] = torch.arange(self.max_cache_size, self.max_cache_size + nids_new.numel(), dtype=self.id_type ,device = self.cache_bucket.device)
        result = self.find_first_indices1(nids_new, step)
        self.cache_bucket[nids_new] = -1
        
        keep_idx = torch.where(result[self.max_cache_size:])[0]
        drop_idx = torch.where(~result[self.Fixed:self.max_cache_size])[0]
        
        if drop_idx.numel() == 0 or keep_idx.numel() == 0:
            return 
        exchange_num = min(drop_idx.numel(), keep_idx.numel())
        if keep_idx.numel() <= drop_idx.numel():
            evict_idx = drop_idx[:exchange_num]
            insert_idx = keep_idx
        else:
            evict_idx = drop_idx
            insert_idx = keep_idx[:exchange_num]
            

        candidates = nids_new[insert_idx]
        evict_nids = self.point[evict_idx]
        self.point[evict_idx] = candidates
        value = self.cache_bucket[evict_nids]
        self.cache_bucket[evict_nids] = -1
        self.cache_bucket[candidates] = value
        self.gpu_fix_cache[value] = feats[insert_idx]
        self.evit_num += exchange_num
    
    def find_first_indices1(self, a: torch.Tensor, step):
        find_mask = torch.full((self.max_cache_size + a.size(0),), fill_value=False, dtype=torch.bool)
        kk = min(self.Visibility - 1 , self.limits - step -1)
        while kk >= 0:
            idx = (step + kk) % (self.Visibility + 1)
            b = self.task_queue[idx*self.avg_input_size:idx *self.avg_input_size+self.offset_list[idx]]
            bucket = self.cache_bucket[b]
            mask = bucket >= self.Fixed
            find_mask[bucket[mask]] = True
            kk -= 1
        return find_mask


    def worker(self, handle_dict, task_queue, signal ,way, ii):
        for i in range(2):
            self.miss_rate = 0.0
            self.gate_rate = 0.0
            while task_queue.empty():
                pass
            item = task_queue.get()
            self.limits= item
            step = 0
            while step < self.limits:
                start_time = time.time()
                idx = step % (self.Visibility+1)
                input_nodes = self.task_queue[idx*self.avg_input_size:idx *self.avg_input_size+self.offset_list[idx]]
                curr_gate_mask = self.gate_mask[idx*self.avg_input_size:idx *self.avg_input_size+self.real_len[idx]]
                cache_mask = self.cache_bucket[input_nodes] != -1
                nids_cpu = input_nodes[~cache_mask]
                start_trans = time.time()
                self.pinned_buffer[:nids_cpu.numel()].copy_(self.ndata[nids_cpu], non_blocking=True)
                self.feature[0:self.real_len[idx]][curr_gate_mask][~cache_mask] = self.pinned_buffer[:nids_cpu.numel()].to(self.device ,non_blocking=True)
                end_trans = time.time()
                nids_cuda = input_nodes[cache_mask]
                self.miss_rate += nids_cuda.numel()/self.real_len[idx]
                on_cache_idx = self.cache_bucket[nids_cuda]
                self.feature[0:self.real_len[idx]][curr_gate_mask][cache_mask] = self.gpu_fix_cache[on_cache_idx]
                self.num_iter += end_trans - start_trans
                handle_dict.put(0 ,block=False)
                start_wait = time.time()
                if step + self.Visibility < self.limits:
                   iitm = signal.get()
                end_wait = time.time()
                self.wait_time += end_wait - start_wait
                if way == 1:
                    self._evict_slots2(nids_cpu, self.feature[0:self.real_len[idx]][curr_gate_mask][~cache_mask], step + 1)
                    self.evict_time += time.time() - end_wait
                if step == 0:
                    self.miss_rate -= nids_cuda.numel()/self.real_len[idx]
                self.gate_rate += self.offset_list[idx]/self.real_len[idx]
                step += 1
                self.time += time.time() - start_time
            self.miss_rate -= nids_cuda.numel()/self.real_len[idx]
            txt_path = f"time_for_{str(self.feat_name)}.log"
            if i%3 == 2:
                with open(txt_path, "a") as f:
                    f.write(f"{i}\n")
                    f.write(f"miss_rate: {self.miss_rate/(self.limits-2)}\n")
                    f.write(f"avg time: {self.time/self.limits}\n")
                    f.write(f"avg wait_time: {self.wait_time/(self.limits)}\n")
                    f.write(f"avg evict_time: {self.evict_time/(self.limits - self.Visibility)}\n")
                    f.write(f"avg trans_time: {self.num_iter/self.limits}\n")
                    f.write(f"gate_rate: {self.gate_rate/(self.limits)}\n")
                    f.write(f"evict_num:{self.evit_num/(self.limits - self.Visibility)}\n")
            self.time = 0.0
            self.wait_time = 0.0
            self.evict_time = 0.0
            self.evit_num = 0
            self.num_iter = 0.0
            self.zuz = 0.0

def load_feats(args ,task_queue = None):
    if args.feat_name == "yelp":
        dataset_name = 'Yelp'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '/data/data', dataset_name) 
        dataset = Yelp(path, transform=T.NormalizeFeatures())
        data = dataset[0]
        paper_feat = data.x
        return paper_feat
    elif args.feat_name == "mag":
        dataset = MAG240MDataset(root="/data")
        paper_feat = torch.from_numpy(dataset.paper_feat)
        return paper_feat
    elif args.feat_name == "IGB260M":
        dataset = IGB260MDGLDataset_part(args , only_feat=True)
        paper_feat = dataset[0][0:100000000]
        return paper_feat
    elif args.feat_name == "paper":
        while task_queue.empty():
            pass
        item = task_queue.get()
        paper_feat = item
        return paper_feat 
    elif args.feat_name == "re":
        data = RedditDataset()
        g = data[0]
        paper_feat = g.ndata["feat"]
        return paper_feat
def cache_worker_main(args ,node_num, gpuid, handle_dict, task_queue, signal):
    import torch
    import dgl
    print("start")
    prefill = ["hotmap", "degree"]
    p = psutil.Process()
    paper_feat = load_feats(args,task_queue = task_queue)
    print(paper_feat.shape)
    p = psutil.Process()
    
    range_list = [0]
    cores_socket_1 = list(range(20, 40))
    cores_socket_1.extend(list(range(60, 80)))
    try:
        p.cpu_affinity(cores_socket_1)
    except Exception as e:
        print(f"❌ Failed to set CPU affinity: {e}")
    torch.set_num_threads(40)
    num_threads = "40"
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    cache_server = BlockFeatureCacheServer(node_num , paper_feat,args,gpuid )
    handle1 = cuda_ipc_get.get_cuda_ipc_handle(cache_server.feature)
    handle_b1 = base64.b64encode(handle1)
    while task_queue.empty():
            pass
    item = task_queue.get()
    cache_server.prefill = item
    handle_dict.put((handle_b1 ,cache_server.feature.shape))
    while task_queue.empty():
        pass
    item = task_queue.get()
    cache_server.task_queue , cache_server.offset_list ,cache_server.gate_mask ,cache_server.real_len = item
    
    for ii in range_list:
          if ii == 0:
              cache_server.prefill_by_hotmap()
          elif ii == 1:
              cache_server.prefill_by_degree()
          cache_server.worker(handle_dict, task_queue, signal , 0 ,prefill[ii])
