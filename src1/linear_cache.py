from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
from .linear_wrapper import QwenLinearWrapper
from .expert_wrapper import QwenExpertWrapper
from safetensors.torch import safe_open
import torch
from torch import nn
import os
import random
import json

ExpertUID = Any


@dataclass
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool            # 核心标识：指“残差(b)”是否未被加载到 GPU 动态执行池
    offloaded_index: int       # 在 CPU RAM (Pinned Memory) 中的固定位置
    main_index: int = 0        # 在 GPU 动态执行缓冲池中的位置
    prefetched: bool = False
    buff_id = -1
    status: str = field(default="in_ram")
    high_prec : bool = False
    
    # --- 💡 新增：追踪低精度 Base(a) 的独立状态 ---
    base_in_gpu: bool = False  # 低精度 Base(a) 是否已加入 GPU 常驻池
    base_gpu_index: int = -1   # 若常驻 GPU，其在常驻池中的物理索引


@dataclass
class EvictionGroupInfo:
    # 这里的 LRU 逻辑保持原样即可，因为驱逐的粒度现在变成了“动态执行缓冲块(Buffer)”
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)  #标识当前GPU可用专家
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            return info  # least recently used
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class LinearCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, 
                 base_gpu_size: int, # 💡 新增：GPU可常驻多少个 Base(a)
                 ):
        
        self.module_type = self.w1_size = self.w2_size = self.w3_size = self.device = None
        self.active = False
        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        self.main_size = main_size
        self.buff_size = buffer_size
        self.base_in_gpu_size = base_gpu_size
        dummy_module = self._check_module(make_module())

        # 1. 动态执行缓冲池 (GPU) - 包含 a_buffer 和 b_buffer
        self.main_modules = [self._check_module(make_module()) for _ in range(main_size+base_gpu_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size+base_gpu_size)]

        # 2. 全量内存池 (CPU Pinned RAM) - 容纳所有的 a 和 b
        assert self.w1_size is not None
        self.w1_a_ram = [torch.UntypedStorage(self.w1_size).pin_memory() for _ in range(offload_size)]
        self.w1_b_ram = [torch.UntypedStorage(self.w1_size).pin_memory() for _ in range(offload_size)]
        self.w2_a_ram = [torch.UntypedStorage(self.w2_size).pin_memory() for _ in range(offload_size)]
        self.w2_b_ram = [torch.UntypedStorage(self.w2_size).pin_memory() for _ in range(offload_size)]
        
        self.global_zero_gate_up = torch.full((self.w1_size,), fill_value=240, dtype=torch.uint8, device=self.device)
        self.global_zero_down = torch.full((self.w2_size,), fill_value=240, dtype=torch.uint8, device=self.device)
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]
      
        # 3.1 💡 常驻显存池 (GPU) - 专门用于存放不会被驱逐的 Base(a)
        self.base_gpu_free_indices = list(range(base_gpu_size))
        self.w1_a_gpu = [torch.UntypedStorage(self.w1_size,device=self.device) for _ in range(base_gpu_size)]
        self.w2_a_gpu = [torch.UntypedStorage(self.w2_size,device=self.device) for _ in range(base_gpu_size)]
        self.w1_b_gpu = [torch.UntypedStorage(self.w1_size,device=self.device) for _ in range(4)]
        self.w2_b_gpu = [torch.UntypedStorage(self.w2_size,device=self.device) for _ in range(4)]
        self.b_indices = list(range(4))
        # 3.2 💡 临时缓冲区 (GPU) - 专门用于存放每一层缓存miss的Base(a)和residual
        self.base_gpu_buffer_indices = list(range(main_size))
        self.w1_a_gpu_buffer = [torch.UntypedStorage(self.w1_size,device=self.device) for _ in range(buffer_size)]
        self.w1_b_gpu_buffer = [torch.UntypedStorage(self.w1_size,device=self.device) for _ in range(main_size)]
        self.w2_a_gpu_buffer = [torch.UntypedStorage(self.w2_size,device=self.device) for _ in range(buffer_size)]
        self.w2_b_gpu_buffer = [torch.UntypedStorage(self.w2_size,device=self.device) for _ in range(main_size)]
        self.device_expert_buffers = deque([])
        self.map_buffer_posi = {}
        for i in range(buffer_size):
            buffer_module = self._check_module(make_module())
            self.device_expert_buffers.append((buffer_module,i))
        self.info2buffer = {}
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)
        self.demand_stream = torch.cuda.Stream(priority=-1)  # 用于 _load (按需加载)
        self.prefetch_stream = torch.cuda.Stream(priority=0) # 用于 prefetch (后台预取)
        self.prefetch_lock = torch.cuda.Event()
        self.prefetch_uid = None
        self.prefetching = False
        self.interrupt_prefetch_signal = True  # 用于在紧急情况下中断预取任务的派发
        

    def _check_module(self, module):
        # 假设现在的 Wrapper 内包含分离的 w1_a(Base) 和 w1_b(Residual)
        if self.module_type is None:
            self.w1_size = len(module.gate_up_base.storage())
            self.w2_size = len(module.down_base.storage())
            self.device = module.gate_up_base.storage().device
        return module

    def add_expert(self, uid: ExpertUID, gate_up_tensor: torch.Tensor,gate_up_residual_tensor: torch.Tensor, down_tensor: torch.Tensor, down_residual_tensor: torch.Tensor,
                   eviction_group: int = 0, offload: Optional[bool] = None):
        """Register an expert to the cache by directly passing its weight tensors"""
        
        # 确保传入的是连续内存 (contiguous)，否则提取 storage 可能会越界或错位
        gate_up_tensor = gate_up_tensor.contiguous()
        down_tensor = down_tensor.contiguous()
        
        # 直接扒出 Tensor 底层的物理显存/内存块 (Storage) 并送入下一环
        # 注意：在较新的 PyTorch 中，推荐使用 untyped_storage() 替代原有的 .storage()
        return self.add_expert_storages(
            uid, 
            [gate_up_tensor.untyped_storage(), gate_up_residual_tensor.untyped_storage()], [down_tensor.untyped_storage(), down_residual_tensor.untyped_storage()],
            eviction_group=eviction_group, 
        )
    
    def register_ssd_expert(self, uid: ExpertUID, eviction_group: int):
        """
        逻辑注册：在 Cache 中备案，但不分配 RAM 空间。
        """
        assert uid not in self.registered_experts, f"专家 {uid} 已注册"
        
        # 1. 构造 SSD 元数据
        # 这里的 Key 规则要对应 Qwen3 的 Stacked 存储格式

        # 2. 创建 Info 对象
        info = ExpertInfo(
            uid=uid,
            eviction_group=eviction_group,
            offloaded=True,
            status="on_disk",      # 初始状态设为硬盘
        )

        # 3. 注册到全局索引
        self.registered_experts[uid] = info
        self.group_infos[eviction_group].add(info)


    def _get_available_ram_slot(self):
        return random.randint(0, len(self.w1_ram_storages) - 1)  # 简单随机分配，实际可以更智能地选择空位或LRU踢出

    def add_expert_storages(self, uid: ExpertUID, a_storages: List[torch.UntypedStorage], b_storages: List[torch.UntypedStorage], eviction_group: int = 0):
        """
        装载逻辑：全部进入 RAM，尽最大努力把 a 塞进 GPU 常驻池
        """
        assert uid not in self.registered_experts
        
        # 1. 寻找 CPU RAM 空位
        ram_idx = next(i for i, info in enumerate(self.offloaded_infos) if info is None)
        
        # 2. 拷贝至 Pinned Memory
        self.w1_a_ram[ram_idx].copy_(a_storages[0])
        self.w1_b_ram[ram_idx].copy_(a_storages[1])
        self.w2_a_ram[ram_idx].copy_(b_storages[0])
        self.w2_b_ram[ram_idx].copy_(b_storages[1])

        info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, offloaded_index=ram_idx)

        # 3. 💡 尝试放入 GPU 常驻 Base 池
        if len(self.base_gpu_free_indices) > 0:
            gpu_idx = self.base_gpu_free_indices.pop(0)
            self.w1_a_gpu[gpu_idx].copy_(a_storages[0])
            self.w2_a_gpu[gpu_idx].copy_(b_storages[0])
            self.main_modules[gpu_idx].point_to(
                self.w1_a_gpu[gpu_idx],
                self.global_zero_gate_up,
                self.w2_a_gpu[gpu_idx],
                self.global_zero_down,
            )
            info.high_prec = False
            info.offloaded = False
            info.base_in_gpu = True
            info.main_index = gpu_idx
            info.base_gpu_index = gpu_idx
        elif len(self.base_gpu_buffer_indices) > 0:
            gpu_idx = self.base_gpu_buffer_indices.pop(0)
            self.w1_a_gpu_buffer[gpu_idx].copy_(a_storages[0])
            self.w2_a_gpu_buffer[gpu_idx].copy_(b_storages[0])
            idx_in_main_modules = gpu_idx + self.base_in_gpu_size
            self.main_modules[idx_in_main_modules].point_to(
                self.w1_a_gpu_buffer[gpu_idx],
                self.global_zero_gate_up,
                self.w2_a_gpu_buffer[gpu_idx],
                self.global_zero_down,
            )
            info.buff_id = gpu_idx
            info.offloaded = False
            info.high_prec = True
            info.main_index = idx_in_main_modules
            self.group_infos[eviction_group].add(info)
        else:
            self.group_infos[eviction_group].add(info)
        self.registered_experts[uid] = self.offloaded_infos[ram_idx] = info

                
    def check(self, layer_index, selected_experts):
        #check whether the selected_experts in layer layer_index are in the cache
        uids_list = []
        for expert in selected_experts:
            uid = (layer_index, expert)
            if self.registered_experts[uid].offloaded:
                uids_list.append(uid)
        return uids_list if len(uids_list) > 0 else None

    def release(self,uids):
        for uid in uids:
            info = self.registered_experts[uid]
            if info.prefetched:
                self.info2buffer[info.uid][0].free = True
                del self.info2buffer[info.uid]
                info.prefetched = False
                info.offloaded = True


    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False ,high_prec_experts: set = set()) -> Iterator[Tuple[ExpertUID, QwenExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]  # 为每一层划定一个独立的“显存地盘”和“淘汰策略
        for info in infos:
            eviction_group.mark_used(info)  #LRU,保护最近要用的专家不会被淘汰

        try:
            self.active = True
            pre_loaded_infos = deque([])
            infos_to_load = deque([])
            # save pre-loaded experts before they can be swapped
            for info in infos:
                if not info.offloaded:
                   if (info.uid[1] in high_prec_experts and not info.high_prec):
                       print(f"Expert {info.uid} is currently low-precision in GPU but is needed as high-precision.")
                       infos_to_load.append(info)
                   else:
                       pre_loaded_infos.append(info)
                else:
                    infos_to_load.append(info) 
            #pre_loaded_experts = deque([self.main_modules[info.main_index] for info in pre_loaded_infos])
            pre_loaded_experts = deque([])
            for info in pre_loaded_infos:
                if info.prefetched:
                    info_to_evict = eviction_group.choose_expert_to_evict()
                    self._swap(info, info_to_evict)
                    info.buff_id = info_to_evict.buff_id
                    pre_loaded_experts.append(self.main_modules[info.main_index])
                else:
                    info.buff_id = info.main_index
                    pre_loaded_experts.append(self.main_modules[info.main_index])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            window_size = min(len(self.device_expert_buffers) - 1,
                              len(eviction_group.main_infos),
                              len(infos_to_load))
            if window_size == 0 and len(infos_to_load) > 0:
                print("\n" + "="*50, flush=True)
                print("[FATAL] 流水线卡死，window_size 为 0！排查清单：", flush=True)
                print(f"1. (Buffer数 - 1) = {len(self.device_expert_buffers) - 1}", flush=True)
                print(f"2. 常驻池容量 (main_infos) = {len(eviction_group.main_infos)}", flush=True)
                print(f"3. 待加载专家数 = {len(infos_to_load)}", flush=True)
                print("="*50 + "\n", flush=True)
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._load(info_to_load, eviction_group.choose_expert_to_evict() ,info_to_load.uid[1] in high_prec_experts))
            
            if self.prefetch_uid is not None:
                self.prefetching=True
                prefeatch_infos = [self.registered_experts[uid] for uid in self.prefetch_uid]
                self.prefetch(prefeatch_infos)

            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    yield (info.uid, experts_in_loading.popleft())
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._load(info_to_load, eviction_group.choose_expert_to_evict(), info_to_load.uid[1] in high_prec_experts))
                else:
                    if len(pre_loaded_infos) > 0 and info is not pre_loaded_infos[0]:
                        print(f"[Warning] Expert {info.uid} is pre-loaded but not in expected order. This may indicate a logic error.")
                    if len(infos_in_loading) > 0 and info is not infos_in_loading[0]:
                        print(f"[Warning] Expert {info.uid} with status {info.offloaded,info.prefetched} is loading but not in expected order. This may indicate a logic error.")
                    if len(infos_in_loading) == 0 and not info.offloaded == False:
                       print(f"Debug: buffer_len={len(self.device_expert_buffers)}, main_infos={len(eviction_group.main_infos)}")
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False
        


    def prefetch(self, infos_to_load: List[ExpertInfo]):
        # ==========================================
        # 1. 任务拆分与 Buffer 分配 (在 CPU 端瞬间完成)
        # ==========================================
        base_tasks = []      # 阶段1：所有专家的低精度 Base 块
        residual_tasks = []  # 阶段2：需要高精度的专家的 Residual 块

        for index_load in range(len(infos_to_load)):
            info_to_load = infos_to_load[index_load]
            if not info_to_load.base_in_gpu:
                # 申请 Buffer
                # 💡 软件级信号量检测 (防止队列积压过深)
                # 如果此时模型主线发现预测错误严重，修改了此信号，直接放弃后续任务的派发
                if getattr(self, "interrupt_prefetch_signal", False):
                    break 
                item = self.device_expert_buffers.popleft()
                device_expert_buffer, buffer_id = item[0], item[1]
                device_expert_buffer.free = False
                # 更新注册表状态 (抢占坑位)
                info_to_load.prefetched = True
                info_to_load.offloaded = False
                info_to_load.high_prec = False
                self.info2buffer[info_to_load.uid] = (device_expert_buffer, buffer_id)
                self.device_expert_buffers.append((device_expert_buffer, buffer_id))

                # 任务归类
                base_tasks.append((info_to_load, device_expert_buffer, buffer_id))
        # ==========================================
        # 2. 发送物理传输指令 (硬件自动调度与抢占)
        # ==========================================
        # 🌟 使用低优先级流，一旦 demand_stream 启动，硬件会自动压缩这里的 PCIe 带宽
        with torch.cuda.stream(self.prefetch_stream):
            
            # --- 阶段 A：突发传输所有低精度 Base (保证最底线的可用性) ---
            for info, buf, buf_id in base_tasks:
                self.w1_a_gpu_buffer[buf_id].copy_(self.w1_a_ram[info.offloaded_index], non_blocking=True)
                self.w2_a_gpu_buffer[buf_id].copy_(self.w2_a_ram[info.offloaded_index], non_blocking=True)
                buf.point_to(
                    self.w1_a_gpu_buffer[buf_id], 
                    self.global_zero_gate_up,  # 预取阶段先不管 Residual，直接指向全零块
                    self.w2_a_gpu_buffer[buf_id],
                    self.global_zero_down,
                )
        # 记录 Event 锁，确保计算前这些东西已经拉取完毕
        self.prefetch_lock.record(self.prefetch_stream)
    
        
          
    def _load(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo ,need_high: bool) -> nn.Module:
        """主加载流 (阻塞/同步 fallback)，逻辑与 prefetch 完全同构"""
        if info_to_load.offloaded:
            item = self.device_expert_buffers.popleft()
            device_expert_buffer = item[0]
            buffer_id = item[1]
            with torch.cuda.stream(self.demand_stream):
                # 2. 条件拷贝或指向 a
                    # 1. 拷贝 b
                self.w1_a_gpu_buffer[buffer_id].copy_(self.w1_a_ram[info_to_load.offloaded_index], non_blocking=True)
                self.w2_a_gpu_buffer[buffer_id].copy_(self.w2_a_ram[info_to_load.offloaded_index], non_blocking=True)
                info_to_load.high_prec = False
            if need_high:
                self.w1_b_gpu_buffer[info_to_evict.buff_id].copy_(self.w1_b_ram[info_to_load.offloaded_index], non_blocking=True)
                self.w2_b_gpu_buffer[info_to_evict.buff_id].copy_(self.w2_b_ram[info_to_load.offloaded_index], non_blocking=True)
                info_to_load.high_prec = True
            device_expert_buffer.point_to(
                    self.w1_a_gpu_buffer[buffer_id],
                    self.w1_b_gpu_buffer[info_to_evict.buff_id] if need_high else self.global_zero_gate_up,
                    self.w2_a_gpu_buffer[buffer_id],
                    self.w2_b_gpu_buffer[info_to_evict.buff_id] if need_high else self.global_zero_down,
                )
            self.device_expert_buffers.append((self.main_modules[info_to_evict.main_index], buffer_id))
            info_to_evict.high_prec = False
            self.main_modules[info_to_evict.main_index] = device_expert_buffer
            # rm device_expert_buffer from self.device_expert_buffers
            self.main_infos[info_to_evict.main_index] = info_to_load
            info_to_evict.offloaded, info_to_load.offloaded = True, False
            info_to_load.buff_id = info_to_evict.buff_id
            info_to_load.main_index = info_to_evict.main_index
            self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
            return device_expert_buffer
        else:
            with torch.cuda.stream(self.demand_stream):
                self.w1_b_gpu[info_to_load.buff_id].copy_(self.w1_b_ram[info_to_load.offloaded_index])
                self.w2_b_gpu[info_to_load.buff_id].copy_(self.w2_b_ram[info_to_load.offloaded_index])
                device_expert_buffer = self.main_modules[info_to_load.main_index]
                if info_to_load.base_in_gpu:
                    device_expert_buffer.point_to(
                        self.w1_a_gpu[info_to_load.base_gpu_index], self.w1_b_gpu[info_to_load.buff_id],
                        self.w2_a_gpu[info_to_load.base_gpu_index], self.w2_b_gpu[info_to_load.buff_id],
                    )
                else:
                    device_expert_buffer.point_to(
                        self.w1_a_gpu_buffer[info_to_load.main_index], self.w1_b_gpu[info_to_load.buff_id],
                        self.w2_a_gpu_buffer[info_to_load.main_index], self.w2_b_gpu[info_to_load.buff_id],
                    )
            return device_expert_buffer
    
    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
         ### bug! need to remove the buffer from  the buffers
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""

        device_expert_buffer = self.info2buffer[info_to_load.uid][0]
        buffer_id = self.info2buffer[info_to_load.uid][1]
        device_expert_buffer.free = True
        self.device_expert_buffers.append((self.main_modules[info_to_evict.main_index], buffer_id))
        self.main_modules[info_to_evict.main_index] = device_expert_buffer
        # rm device_expert_buffer from self.device_expert_buffers
        self.device_expert_buffers.remove((device_expert_buffer, buffer_id))
        self.main_infos[info_to_evict.main_index] = info_to_load
        info_to_evict.offloaded = True
        info_to_load.main_index = info_to_evict.main_index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        del self.info2buffer[info_to_load.uid]
        info_to_load.prefetched = False
        return device_expert_buffer