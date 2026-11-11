import torch
import dgl
from collections import deque
from typing import Dict, Optional, List
import numpy as np
import cupy as cp
import time
import argparse
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
import time
import tqdm
import argparse
import pandas as pd
import dgl
import numpy as np
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler
import dgl.nn.pytorch as dglnn
from igb_dataloader import IGB260MDGLDataset_part



class BlockFeatureCacheServer:
    def __init__(self, graph, gpuid: Optional[int], feature_field: str = 'feat', label_field: str = 'label', window_size: int = 3):
        self.graph = graph
        self.gpuid = gpuid
        self.feature_field = feature_field
        self.label_field = label_field
        self.node_num = graph.num_nodes()
        self.full_cached = False
        self.max_cache_size = int(self.node_num * 0.40)
        self.device = torch.device('cuda', self.gpuid) if self.gpuid is not None else torch.device('cpu')
        self.prefill_num = int(self.max_cache_size * 0.5)
        self.Visibility = 10
        self.limits = 400
        self.Fixed = int(self.max_cache_size * 0.40)
        
        self.gpu_flag = torch.zeros(self.node_num, dtype=torch.bool)
        self.cached_num = 0
        self.capability = self.node_num
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=torch.long,
        )
        self.task_queue = torch.empty(0, dtype=torch.int32)
        self.point = torch.empty(0, dtype=torch.long)
        self.point_position = torch.empty(0, dtype=torch.long)
        self.offset_list = []
        self.cache_capacity = 0
        self.find_time = 0.0
        self.sort_time = 0.0
        self.exchange_time = 0.0
        self.degree = self.graph.out_degrees()

    def prefill_by_degree(self):
        self.Fixed = int(self.max_cache_size * 0.40)
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=torch.long,
        )
        # self.gpu_fix_cache = torch.empty(
        #     (self.max_cache_size, self.feat_dim),
        #     dtype=self.graph.ndata[self.feature_field].dtype,
        #     device=self.device
        # )
        try:
            degrees = self.graph.in_degrees()
        except Exception:
            degrees = torch.ones(self.node_num, dtype=torch.long)
        topk_vals, topk_idx = torch.topk(degrees, k=self.max_cache_size, largest=True)
        topk_idx = topk_idx.long()
        slots = torch.arange(0, self.max_cache_size, dtype=torch.long)
        # feats = self.graph.ndata[self.feature_field][topk_idx].to(self.device, non_blocking=True)
        self.point = topk_idx[self.Fixed:]
        # self.gpu_fix_cache[slots.to(self.device)] = feats
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.time = 0.0

    def prefill_by_random(self):
        self.Fixed = int(self.max_cache_size * 0.40)
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=torch.long,
        )
        # self.gpu_fix_cache = torch.empty(
        #     (self.max_cache_size, self.feat_dim),
        #     dtype=self.graph.ndata[self.feature_field].dtype,
        #     device=self.device
        # )
        topk_idx = torch.randint(0, self.node_num, (self.max_cache_size,), dtype=torch.long)
        topk_idx = topk_idx.long()
        slots = torch.arange(0, self.max_cache_size, dtype=torch.long)
        # feats = self.graph.ndata[self.feature_field][topk_idx].to(self.device, non_blocking=True)
        self.point = topk_idx[self.Fixed:]
        # self.gpu_fix_cache[slots.to(self.device)] = feats
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.time = 0.0

    def prefill_by_hot_map(self ,train_dataloader,hot_map = None):
        self.Fixed = int(self.max_cache_size * 0.50)
        self.cache_bucket = torch.full(
            (self.node_num,),
            fill_value=-1,
            dtype=torch.long,
        )
        if hot_map is None:
            hot_map = torch.zeros(self.node_num, dtype=torch.long)
            dataloader_iter = iter(train_dataloader)
            for i in range(len(train_dataloader)):
                input_nodes, output_nodes, blocks = next(dataloader_iter)
                hot_map[input_nodes] += 1
        topk_vals, topk_idx = torch.topk(hot_map, k=self.max_cache_size, largest=True)
        slots = torch.arange(0, self.max_cache_size, dtype=torch.long)
        self.point = topk_idx[self.Fixed:]
        self.cache_bucket[topk_idx] = slots
        self.miss_rate = 0.0
        self.time = 0.0
    def _evict_slots(self, nids_new, step):
        if nids_new.numel() == 0:
            return
        self.cache_bucket[nids_new] = torch.arange(self.max_cache_size, self.max_cache_size + nids_new.numel(), dtype=torch.long)
        find_mask, result = self.find_first_indices(nids_new, step)
        self.cache_bucket[nids_new] = -1
        keep_idx = torch.where(find_mask[self.max_cache_size-self.Fixed:])[0]
        drop_idx = torch.where(~find_mask[:self.max_cache_size-self.Fixed])[0]
        if drop_idx.numel() == 0 or keep_idx.numel() == 0:
            return
        exchange_num = min(drop_idx.numel(), keep_idx.numel())
        evict_idx = drop_idx[:exchange_num]
        if keep_idx.numel() <= drop_idx.numel():
            insert_idx = keep_idx
        else:
            min_vals,insert_idx = torch.topk(result[self.max_cache_size-self.Fixed:], k=exchange_num, largest=False , sorted=False)

        candidates = nids_new[insert_idx]
        evict_nids = self.point[evict_idx]
        self.point[evict_idx] = candidates
        
        value = self.cache_bucket[evict_nids]
        self.cache_bucket[evict_nids] = -1
        self.cache_bucket[candidates] = value
    
    def find_first_indices(self, a: torch.Tensor, step):
        result = torch.full((self.max_cache_size-self.Fixed + a.size(0),), fill_value=self.Visibility, dtype=torch.long)
        find_mask = torch.full((self.max_cache_size-self.Fixed + a.size(0),), fill_value=False, dtype=torch.bool)
        kk = self.Visibility-1
        while kk >= 0:
            if step + kk + 1 >= self.limits:
                kk -= 1
                continue
            b = self.task_queue[self.offset_list[step + kk]:self.offset_list[step + kk + 1]]
            bucket = self.cache_bucket[b]
            mask = bucket >= self.Fixed
            bucket_ids = bucket[mask] - self.Fixed
            find_mask[bucket_ids] = True
            result[bucket_ids] = kk
            kk -= 1
        return find_mask,result

    def worker(self,way_to_evict):
        step = 0
        while step < self.num_iter:
            input_nodes = self.task_queue[self.offset_list[step]:self.offset_list[step + 1]]
            on_cache_mask = self.cache_bucket[input_nodes] != -1
            nids_cpu = input_nodes[~on_cache_mask]
            nids_cuda = input_nodes[on_cache_mask]
            self.miss_rate += nids_cpu.numel()/input_nodes.numel()
            on_cache_idx = self.cache_bucket[nids_cuda]
            if step == 0 or step == self.num_iter-1:
                self.miss_rate -= nids_cpu.numel()/input_nodes.numel()
            if way_to_evict == 1:
                self._evict_slots(nids_cpu, step + 1)
            step += 1
        
        
def get_iter(train_dataloader):
        dataloader_iter = iter(train_dataloader)
        num = len(train_dataloader)
        offset_list= [0]
        result_nodes = torch.empty(num*700000, dtype=torch.long)
        for i in range(num):
            input_nodes, output_nodes, blocks = next(dataloader_iter)
            result_nodes[offset_list[i]:offset_list[i]+input_nodes.numel()] = input_nodes
            offset_list.append(offset_list[i]+input_nodes.numel())
        return result_nodes[:offset_list[-1]], offset_list
        
        

def cache_worker_main(gpuid):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    
    # Loading dataset
    parser.add_argument('--path', type=str, default='/data/IGB-dataset/', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='large',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--sample', type=str, default="nei",
        choices=["random_walk", "nei" , "weighted"])
    parser.add_argument('--sample_ratio', type=float, default=0.1,
        help='sampling ratio for train/val/test sets (default: 0.01)')


    args = parser.parse_args()
    print("Loading data...")
    dataset = IGB260MDGLDataset_part(args , load_feat=False)
    

    g = dataset[0]
    if type(g) == list:
        g = g[0]
    cache_server = BlockFeatureCacheServer(g, gpuid)
    test_way = ["random","degree" , "hot",  "PreFC-d" ,"PreFC-h"]
    test_cache_radio = [0.005,0.007,0.009,0.011,0.013,0.015 ,0.02 , 0.024 , 0.028, 0.032, 0.036, 0.04, 0.044, 0.048]
    miss_random = []
    miss_degree = []
    miss_hot_map = []
    miss_PreFC_d = []
    miss_PreFC_h = []
    original_train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    
    train_sample_size = int(len(original_train_idx) * 0.01)
    
    train_perm = torch.randperm(len(original_train_idx))
    train_idx = original_train_idx[train_perm[:train_sample_size]]
    
    
    sampler = NeighborSampler(
        [5, 10, 15],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        #prefetch_labels=["label"],
    )
    
    train_dataloader = DataLoader(
            g,
            train_idx,
            sampler,
            device="cpu",
            batch_size=2048,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            # use_uva=use_uva,
        )
    cache_server.task_queue,cache_server.offset_list = get_iter(train_dataloader)
    cache_server.num_iter = len(train_dataloader)
    cache_server.limits = len(train_dataloader)
    num_iter = len(train_dataloader)
    hot_map = torch.zeros(g.num_nodes(), dtype=torch.long)
    dataloader_iter = iter(train_dataloader)
    for i in range(len(train_dataloader)):
        input_nodes, output_nodes, blocks = next(dataloader_iter)
        hot_map[input_nodes] += 1
    for radio in test_cache_radio:
        cache_server.max_cache_size = int(g.num_nodes() * radio)
        for ii in range(0,len(test_way)):
            if test_way[ii] == "random":
                cache_server.prefill_by_random()
                cache_server.worker(0)
                miss_random.append(cache_server.miss_rate/(num_iter-2))
            elif test_way[ii] == "degree":
                cache_server.prefill_by_degree()
                cache_server.worker(0)
                miss_degree.append(cache_server.miss_rate/(num_iter-2))
            elif test_way[ii] == "PreFC-d":
                cache_server.prefill_by_degree()
                cache_server.worker(1)
                miss_PreFC_d.append(cache_server.miss_rate/(num_iter-2))
            elif test_way[ii] == "hot":
                cache_server.prefill_by_hot_map(train_dataloader , hot_map)
                cache_server.worker(0)
                miss_hot_map.append(cache_server.miss_rate/(num_iter-2))
            elif test_way[ii] == "PreFC-h":
                cache_server.prefill_by_hot_map(train_dataloader , hot_map)
                cache_server.worker(1)
                miss_PreFC_h.append(cache_server.miss_rate/(num_iter-2))
    with open("missrate_compare_igb.txt", "a") as f:
        f.write(f"test_cache_radio: {test_cache_radio}\n")
        f.write(f"miss_random: {miss_random}\n")
        f.write(f"miss_degree: {miss_degree}\n")    
        f.write(f"miss_hot_map: {miss_hot_map}\n")
        f.write(f"miss_PreFC_d: {miss_PreFC_d}\n")
        f.write(f"miss_PreFC_h: {miss_PreFC_h}\n")
   
if __name__ == "__main__":
    cache_worker_main(0)