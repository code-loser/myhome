from collections import defaultdict
import os
import json
from functools import cache
from dataclasses import dataclass
import typing as tp

import torch
from torch import nn

from transformers.models.mixtral import MixtralForCausalLM, MixtralConfig
from transformers import AutoConfig, AutoModelForCausalLM # 使用通用加载器
from transformers.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange

from hqq.core.quantize import BaseQuantizeConfig

from .expert_cache import ExpertCache
from .linear_cache import LinearCache
from .expert_wrapper import QwenExpertWrapper
from .linear_wrapper import QwenLinearWrapper
from .custom_layers import (
    HQQLinearTritonSavable,
    MixtralBLockSparseTop2MLP_HQQ,
    SparseMoeWrapper,
    QwenMoeBlock_NoQuant
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoePreTrainedModel
from .utils import with_default_dtype
import gc
from accelerate import init_empty_weights
from safetensors.torch import safe_open
import json


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    cache_strategy: list



def load_00_expert_state_dict(states_dir: str, device: torch.device):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.0.block_sparse_moe.experts.0"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))


def build_model(device, offload_config, state_path):
    # 1. 加载配置
    config = AutoConfig.from_pretrained(state_path, trust_remote_code=True)
    # Qwen3-VL 通常嵌套在 language_model 字段下
    model_config = getattr(config, "text_config", config)
    num_layers = model_config.num_hidden_layers
    num_experts = model_config.num_experts
    # 2. 构造“骨架”模型 (Meta Device)
    with init_empty_weights():
        # 确保类名匹配，若不匹配请替换为实际的类
        model = Qwen3VLMoeForConditionalGeneration(config)
    shape_gate_up = (2048, 1536)
    shape_down = (768, 2048)
    # 3. 初始化专家缓存
    def _make_module():
        return QwenExpertWrapper(shape_gate_up, shape_down, device)
    
    # 1. 加载索引地图
    index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
        weight_map = index_data["weight_map"] # 这是一个 key: filename 的字典

    # 2. 定义一个句柄缓存，避免重复打开文件
    handle_cache = {}

    def get_handle(filename):
        if filename not in handle_cache:
            file_path = os.path.join(state_path, filename)
            handle_cache[filename] = safe_open(file_path, framework="pt", device="cpu")
        return handle_cache[filename]

    expert_cache = LinearCache(
        make_module=_make_module,
        main_size=8,
        offload_size=48*128,
        buffer_size=18,
        base_gpu_size=0,
    )

    # 4. 准备加载权重 (使用 safe_open 避免 RAM 爆炸)
    # 假设权重在单个文件或分片文件中，此处以单文件为例
    # 若是多文件，需遍历多个 safetensors
    
    shard_to_non_expert_keys = defaultdict(list)

    for key, shard_name in weight_map.items():
        if "mlp.experts" not in key:
            shard_to_non_expert_keys[shard_name].append(key)

    # 2. 按分片进行“批量”加载
    for shard_name, keys in shard_to_non_expert_keys.items():
        shard_path = os.path.join(state_path, shard_name)
        
        # 使用 safe_open 打开当前分片
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                if key == "lm_head.weight":
                    print(f"正在加载 {key},类型为{tensor.dtype}")
                if tensor.is_floating_point():
                   tensor = tensor.to(torch.bfloat16)
                try:
                    set_module_tensor_to_device(
                        model, 
                        key, 
                        device=device, 
                        value=tensor,
                    )
                except AttributeError as e:
                    print(f"[Warning] Failed to set {key} as {key}: {e}")
        
        # 每加载完一个分片，手动清理，防止 RAM 堆积
        gc.collect()
    for m in model.modules():
        # named_buffers(recurse=False) 只获取当前 module 直接拥有的 buffer
        # 这样返回的 name 就不包含 "." 了
        for name, buf in m.named_buffers(recurse=False):
            if buf is not None and buf.device.type != device:
                # 重新注册到正确的设备
                # 注意：有些模型实现里 inv_freq 是不需要 persistent 的
                target_dtype = torch.bfloat16 if buf.is_floating_point() else buf.dtype
                # 重新注册到正确的设备和精度
                m.register_buffer(name, buf.to(device=device, dtype=target_dtype), persistent=False)
    print("Non-expert parameters (Embedding, Attention, Gate, etc.) loaded successfully.")

    # B. 遍历层，处理专家参数并进行“手术式”替换
    for i in range(num_layers):
        layer = model.model.language_model.layers[i] # 对应你给的 key 路径
        
        # 5. 封装 SparseMoeWrapper 替换原有的 MLP
        # 这里的 gate 已经通过上面的全量加载 move 到 device 了
        new_moe = SparseMoeWrapper(
            model_config,
            layer_id=i,
            gate=layer.mlp.gate, 
            expert_cache=expert_cache,
            layers=model.model.language_model.layers
        )

        base_key = f"model.language_model.layers.{i}.mlp.experts"
        gate_up_key = f"{base_key}.gate_up_proj"
        down_key = f"{base_key}.down_proj"
        gate_up_proj_residual_key = f"{base_key}.gate_up_proj_residual"
        down_proj_residual_key = f"{base_key}.down_proj_residual"
        # 根据索引找到对应的分片文件名
        gate_up_shard = weight_map[gate_up_key]
        down_shard = weight_map[down_key]
        gate_up_residual_shard = weight_map[gate_up_proj_residual_key]
        down_residual_shard = weight_map[down_proj_residual_key]
        # 获取对应的句柄
        h_gate_up = get_handle(gate_up_shard)
        h_down = get_handle(down_shard)
        h_gate_up_residual = get_handle(gate_up_residual_shard)
        h_down_residual = get_handle(down_residual_shard)
        # 获取全层合并的大张量 (注意：如果是超大模型，建议这里也用 get_slice)
        stacked_gate_up = h_gate_up.get_slice(gate_up_key)
        stacked_down = h_down.get_slice(down_key)
        stacked_gate_up_residual = h_gate_up_residual.get_slice(gate_up_proj_residual_key)
        stacked_down_residual = h_down_residual.get_slice(down_proj_residual_key)
        for e_idx in range(num_experts):
            # 1. 纯粹的切片提取：只提取当前专家的权重 Tensor
            # clone() 很重要，它能剥离这块小内存与原始巨型 Storage 的物理绑定
            t_gate_up = stacked_gate_up[e_idx].clone()
            t_down = stacked_down[e_idx].clone()
            t_gate_up_residual = stacked_gate_up_residual[e_idx].clone()
            t_down_residual = stacked_down_residual[e_idx].clone()
            # 2. 直接将 Tensor 喂给 Cache 的底层
            # 没有任何 nn.Module 的实例化开销！
            expert_cache.add_expert(
                uid=(i, e_idx),
                gate_up_tensor=t_gate_up,
                gate_up_residual_tensor=t_gate_up_residual, # 初始残差为0
                down_tensor=t_down,
                down_residual_tensor=t_down_residual, # 初始残差为0
                eviction_group=0,
                offload=True # 初始全部放 CPU/Disk Pinned Memory
            )
        # 🌟 修复 Bug: 整层提取完毕后，才删除全层巨型张量，释放系统 RAM 🌟
        del stacked_gate_up, stacked_down , stacked_gate_up_residual, stacked_down_residual
        model.lm_head.to(torch.bfloat16)
        # 替换层
        layer.mlp = new_moe
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return model

# 辅助函数：将 safetensors 的 key 映射到 torch 对象的属性
def set_module_tensor_to_device(model, tensor_name, device, value):
    # 简单实现，可使用 accelerate.utils.set_module_tensor_to_device
    from accelerate.utils import set_module_tensor_to_device as accelerate_set
    accelerate_set(model, tensor_name, device=device, value=value)