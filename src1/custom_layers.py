import copy
import functools
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Dict, Any
from hqq.core.quantize import HQQLinear, Quantizer

import torch
from torch import nn
from torch.nn import functional as F

from .packing import pack_4bit_u8_common, pack_2bit_u8_common, unpack_4bit_u8_common, unpack_2bit_u8_common
from .triton_kernels import triton_matmul4_transpose, triton_matmul3_transpose, triton_matmul2_transpose
import time
import numpy as np
from collections import defaultdict


class HQQLinearTritonSavable(HQQLinear):
    def __init__(self, layer, quant_config, meta=None, **kwargs):
        """
        Example how to get meta:
        >>>> meta1 = HQQLinearSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
        >>>> meta2 = HQQLinearSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)
        """
        
        assert quant_config['weight_quant_params']['nbits'] in [2, 3, 4]
        
        super().__init__(layer, quant_config, **kwargs)

        
        if not hasattr(self, 'meta'):
            assert meta is not None
            self.meta = copy.deepcopy(meta)
        
        self._register_state_dict_hook(self._add_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_from_state_dict_hook)
    
    def quantize(self, *args, **kwargs):
        super().quantize(*args, **kwargs)
        
        # repacking
        self.repack()
    
    def repack(self):
        if self.W_q.shape != self.meta['shape']:
            W_q = Quantizer.unpack[self.meta['packing']](self.W_q)
            sh = self.meta['shape']
            W_q = W_q.reshape((-1,) + sh[1:])
            W_q = W_q[:sh[0], ...]
            self.W_q = Quantizer.pack[self.meta['packing']](W_q)
    
    def forward(self, x):
        return self.forward_triton(x)
    
    def set_backend(self, backend):
        pass
    
    @torch.inference_mode()
    def forward_triton(self, x):
        assert self.ready, "model was not quantized"
        assert self.meta['axis'] == 0

        W_q, meta = self.W_q, self.meta

        del_keys = []
        if 'quant_scale' in meta and meta['quant_scale']:
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
        if 'quant_zero' in meta and meta['quant_zero']:
            meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')

        K = meta['shape'][1]
        N = meta['shape'][0]
        
        if self.meta['nbits'] == 4:
            fn = triton_matmul4_transpose
        elif self.meta['nbits'] == 3:
            fn = functools.partial(triton_matmul3_transpose, N=N)
        elif self.meta['nbits'] == 2:
            fn = triton_matmul2_transpose
        else:
            raise RuntimeError(f"nbits == {self.meta['nbits']} isn't yet supported")
        
        output = fn(
            meta['group_size'], x,
            W_q.view(-1, K),
            meta['scale'].view(-1, K),
            meta['zero'].view(-1, K),
            bias=self.bias if hasattr(self, 'bias') else None,
        )

        #Cleanup
        for key in del_keys:
            del meta[key]

        return output

    # to support .forward_pytorch(...) - backward compatibility
    @torch.inference_mode()
    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        del_keys = []
        if(meta['quant_scale']):
            meta['scale'] = Quantizer.dequantize(meta['scale_q'], meta['meta_scale']); del_keys.append('scale')
        if(meta['quant_zero']):
            meta['zero']  = Quantizer.dequantize(meta['zero_q'],  meta['meta_zero']);  del_keys.append('zero')
        
        W_q_p = Quantizer.unpack[meta['packing']](W_q).half()
        W_q_p = W_q_p[:meta['shape'][0], ...]
        W_q_p = W_q_p.reshape((meta['group_size'], -1))
    
        if((meta['group_size'] is not None) and (meta['nbits']==3)):
            W_q_p = W_q_p[:meta['group_size']] if (meta['axis']==0) else W_q_p[:,:meta['group_size']]
        W_est = ((W_q_p - meta['zero'])*meta['scale']).reshape(meta['shape']) 
        
        #Cleanup
        del W_q_p
        for key in del_keys: del meta[key]
        return W_est
    
    @classmethod
    def get_hqq_meta(cls, linear_shape, quant_config):
        layer = HQQLinear(nn.Linear(*linear_shape, bias=False), quant_config)
        meta = layer.meta

        def _remove_tensors_recursive(d):
            keys = list(d.keys())

            for k in keys:
                if isinstance(d[k], torch.Tensor):
                    del d[k]
                elif isinstance(d[k], dict):
                    _remove_tensors_recursive(d[k])

        _remove_tensors_recursive(meta)

        return meta
        
    @staticmethod
    def _add_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        tensor_paths = self._get_tensor_paths(self.meta)
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
            'scale', 'zero'}
        )
        
        def _add(name, value):
            state_dict[prefix + name] = value
        
        _add('W_q', self.W_q)
        
        if self.bias is not None:
            _add('bias', self.bias)
        
        if 'meta_scale' in self.meta:
            _add('meta.scale_q', self.meta['scale_q'])
            _add('meta.meta_scale.scale', self.meta['meta_scale']['scale'])
            _add('meta.meta_scale.zero', self.meta['meta_scale']['zero'])
        else:
            _add('meta.scale', self.meta['scale'])
        
        if 'meta_zero' in self.meta:
            _add('meta.zero_q', self.meta['zero_q'])
            _add('meta.meta_zero.scale', self.meta['meta_zero']['scale'])
            _add('meta.meta_zero.zero', self.meta['meta_zero']['zero'])
        else:
            _add('meta.zero', self.meta['zero'])
        
        return state_dict
    
    def _load_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        tensor_paths = [k[len(prefix + 'meta.'):] for k in state_dict.keys() if k.startswith(prefix + 'meta.')]
        assert set(tensor_paths).issubset(
            {'scale_q', 'meta_scale.scale', 'meta_scale.zero', 'zero_q', 'meta_zero.scale', 'meta_zero.zero',
            'scale', 'zero'}
        )
        
        def _del(name):
            del state_dict[prefix + name]
        def _set(name):
            setattr(self, name, state_dict[prefix + name])
            _del(name)
        def _get(name):
            v = state_dict[prefix + name]
            _del(name)
            return v
        
        _set('W_q')
        if 'bias' in state_dict:
            _set('bias')
        else:
            self.bias = None
            
        if not hasattr(self, 'meta'):
            self.meta = {}
        
        if (prefix + 'meta.meta_scale.scale') in state_dict:
            self.meta['scale_q'] = _get('meta.scale_q')
            self.meta['quant_scale'] = True
            if not 'meta_scale' in self.meta:
                self.meta['meta_scale'] = {}
            self.meta['meta_scale'] |= {
                'scale': _get('meta.meta_scale.scale'),
                'zero': _get('meta.meta_scale.zero')
            }
        else:
            self.meta['scale'] = _get('meta.scale')
        if (prefix + 'meta.meta_zero.scale') in state_dict:
            self.meta['zero_q'] = _get('meta.zero_q')
            self.meta['quant_zero'] = True
            if not 'meta_zero' in self.meta:
                self.meta['meta_zero'] = {}
            self.meta['meta_zero'] |= {
                'scale': _get('meta.meta_zero.scale'),
                'zero': _get('meta.meta_zero.zero')
            }
        else:
            self.meta['zero'] = _get('meta.zero')
        self.ready = True
        
        # self.cuda()
        # self.in_gpu = self.W_q.device.type == 'cuda'
        # assert self.in_gpu
        
        self.repack()
        
    @classmethod
    def _get_tensor_paths(cls, state: Dict[str, Any], prefix=''):
        paths = []
        
        for k, v in state.items():
            if isinstance(v, dict):
                paths += cls._get_tensor_paths(v, prefix=k + '.')
            elif isinstance(v, torch.Tensor):
                paths.append(prefix + k)
        
        return paths
    
    def state_dict(self, *args, **kwargs):
        return nn.Module.state_dict(self, *args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        nn.Module.load_state_dict(self, *args, **kwargs)


class MixtralBLockSparseTop2MLP_HQQ(nn.Module):
    def __init__(self, config: MixtralConfig, quant_config: Dict[str, Any], meta1, meta2):
        super().__init__()
        
        self.w1 = HQQLinearTritonSavable(None, quant_config, meta1)
        self.w2 = HQQLinearTritonSavable(None, quant_config, meta2)
        self.w3 = HQQLinearTritonSavable(None, quant_config, meta1)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MarkovExpertPredictorGPU:
    def __init__(self, num_experts=64):
        self.num_experts = num_experts
        # 抛弃 defaultdict(numpy)，直接使用 dict 存储 PyTorch Tensor
        self.transition_counts = {}
        self.transition_probs = {}
        self.layer_stats = {}

    def fit_from_trajectories(self, layer_idx: int, selected_experts: torch.Tensor , top_n: int = 4):
        """
        在 GPU 上极速训练马尔可夫转移矩阵
        :param layer_idx: 当前层 ID
        :param selected_experts: shape (S, K)，例如 (sequence_length, 4)，必须在 GPU 上
        :param top_n: 用于计算覆盖率的前 N 个专家
        """
        S, K = selected_experts.shape
        device = selected_experts.device
        E = self.num_experts

        # 1. 将索引 (S, K) 转换为 Multi-hot 掩码矩阵 (S, E)
        # 例如 Token 0 激活了专家 [2, 5]，则第 0 行在索引 2 和 5 处为 1.0
        multi_hot = torch.zeros((S, E), dtype=torch.float32, device=device)
        multi_hot.scatter_(1, selected_experts, 1.0)

        # 2. 时间步错位对齐
        V_curr = multi_hot[:-1]  # 时刻 t: 取第 0 到 S-2 个 Token
        V_next = multi_hot[1:]   # 时刻 t+1: 取第 1 到 S-1 个 Token

        # 3. 核心魔法：一次矩阵乘法完成所有统计！
        # V_curr.T 形状 (E, S-1), V_next 形状 (S-1, E)
        # 结果 transitions 形状 (E, E)
        transitions = torch.matmul(V_curr.T, V_next)

        total_overlaps = torch.trace(transitions)
        
        # 2. max_possible_overlaps: 理论上最大可能的重合人次
        # 一共有 S-1 对相邻 token，每对 token 最多能重合 K 个专家
        max_possible_overlaps = (S - 1) * K
        
        # 3. overlap_rate: 0.0 到 1.0 之间的标量，直接反映相邻 token 激活的相似度
        overall_overlap_rate = (total_overlaps / max_possible_overlaps).item()
        

        # 4. 累加到全局计数器中
        if layer_idx not in self.transition_counts:
            self.transition_counts[layer_idx] = torch.zeros((E, E), dtype=torch.float32, device=device)
        
        self.transition_counts[layer_idx] += transitions

        # 5. 更新概率矩阵 (行归一化)
        counts = self.transition_counts[layer_idx]
        row_sums = counts.sum(dim=1, keepdim=True)
        # 避免除以 0，将求和为 0 的行分母强行设为 1
        row_sums = torch.clamp(row_sums, min=1e-9) 
        self.transition_probs[layer_idx] = counts / row_sums
        
        probs = self.transition_probs[layer_idx] # 形状: (E, E)

        # 1. 计算平均信息熵 (Shannon Entropy)
        # 加上 1e-9 防止 log(0) 出现 NaN
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        avg_entropy = entropy.mean().item()
        
        # 2. 计算 Top-K 覆盖率 (Top-K Coverage)
        # 如果缓存大小只能容纳 top_n 个专家，能覆盖多少比例的真实路由？
        # topk_probs 形状 (E, top_n)
        topk_probs = torch.topk(probs, top_n, dim=-1)[0]
        avg_topk_coverage = topk_probs.sum(dim=-1).mean().item()

        # 将这些指标记录在类内部，供全局动态调度使用
            
        self.layer_stats[layer_idx] = {
            "avg_entropy": avg_entropy,
            "topk_coverage": avg_topk_coverage,
            "self_transition": overall_overlap_rate
        }
        with open("markov_stats.txt", "a") as f:
            f.write(f"Layer {layer_idx}: Entropy={avg_entropy:.4f}, Top-{top_n} Coverage={avg_topk_coverage:.4f}, Self-Transition={overall_overlap_rate:.4f}\n")

        # 将覆盖率作为指导缓存分配的核心指标返回
        return avg_topk_coverage

    def predict_next_experts(self, layer_idx: int, current_experts: torch.Tensor, top_n: int = 4, n_steps: int = 1):
        """
        在 Decode 阶段全程 GPU 预测下 n_steps 个 Token 的专家，并求出需保留的当前专家。
        
        :param layer_idx: 当前层 ID
        :param current_experts: 当前 Token 选中的专家，shape (K,)，在 GPU 上
        :param top_n: 预测每步最可能用到的前 N 个专家
        :param n_steps: 往前预测的 Token 步数
        :return: (keep_experts, all_predicted_tensor)
                 keep_experts: 交集（当前激活且未来 n_steps 会复用的专家）
                 all_predicted_tensor: 未来 n_steps 涉及到的所有预测专家的并集
        """
        if layer_idx not in self.transition_probs:
            # 冷启动兜底：如果没有统计数据，假设直接复用 current_experts 或者默认前 top_n
            fallback_preds = torch.arange(top_n, device=current_experts.device)
            keep_mask = torch.isin(current_experts, fallback_preds)
            return current_experts[keep_mask], fallback_preds
        
        prob_matrix = self.transition_probs[layer_idx]

        # =================================================================
        # 1. 初始化状态向量 (P1)
        # 这等价于：[当前专家的 Multi-hot 向量] @ prob_matrix
        # state 形状: (E,)
        # =================================================================
        state = prob_matrix[current_experts].sum(dim=0)
        
        all_predicted = []
        
        # =================================================================
        # 2. 模拟马尔可夫状态游走 (GPU 极速矩阵乘法)
        # =================================================================
        for step in range(n_steps):
            # 获取当前步概率最高的 top_n 个专家
            step_preds = torch.topk(state, top_n)[1]
            all_predicted.append(step_preds)
            
            # 如果还需要预测下一步，状态向量向前推演 (P_{t+1} = P_t @ M)
            if step < n_steps - 1:
                state = torch.matmul(state, prob_matrix)
                
        # 将未来 n_steps 预测出的所有专家合并，并去重 (变成一维集合)
        all_predicted_tensor = torch.cat(all_predicted).unique()
        
        # =================================================================
        # 3. 求交集：GPU 原生的 torch.isin
        # 找出当前激活的专家中，有哪些在 future_predictions 里面
        # =================================================================
        intersection_mask = torch.isin(current_experts, all_predicted_tensor)
        keep_experts = current_experts[intersection_mask]
        
        # 建议同时返回这俩：
        # keep_experts 用来“免除淘汰”(Eviction Bypass)
        # all_predicted_tensor 用来“后台预取”(Background Prefetch)
        return keep_experts, all_predicted_tensor

class SparseMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache, layers):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache

        self.layers = layers
        self.threshold = 0
        self.MarkovExpertPredictor = MarkovExpertPredictorGPU(num_experts=self.num_experts)
        self.token_type = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if hidden_states.shape[0]==1 and self.threshold>0:
            if routing_weights[0][1]<self.threshold:
                routing_weights, selected_experts = torch.topk(routing_weights, 1, dim=-1)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if hidden_states.shape[0]>1:
            self.MarkovExpertPredictor.fit_from_trajectories(self.layer_id, selected_experts)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # valid_mask, high_prec_experts, skip_ratio, low_prec_ratio = self.compute_dynamic_expert_routing(
        #     routing_weights, selected_experts, T1=0.6, T2=0.9
        # )
        valid_mask, high_prec_experts, hobb_load, base_load , high_precision_mask = self.compute_dynamic_expert_routing(
            routing_weights, selected_experts, T1=0.0, T2=1.0
        )
        # if hidden_states.shape[0]>1:
        #     with open("debug_moe_ratios.csv", "a", encoding="utf-8") as f:
        #         f.write(f"Layer_{self.layer_id}, {selected_experts.shape}, {hobb_load}, {base_load}, {my_load}\n")
            
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        one_hot_experts = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
        one_hot_experts = one_hot_experts * valid_mask.unsqueeze(-1) 
        expert_mask = one_hot_experts.permute(2, 1, 0)

        active_experts = selected_experts[valid_mask].unique().tolist()
        # if hidden_states.shape[0]>1:
        #     with open("debug_moe_ratios2.csv", "a", encoding="utf-8") as f:
        #         f.write(f"Layer_{self.layer_id}, {len(active_experts)},{(len(active_experts) - len(high_prec_experts))/len(active_experts) if len(active_experts) > 0 else 0}\n")
        #get the unselected experts
        unselected_experts = [i for i in range(self.num_experts) if i not in active_experts]
        self.experts.release([(self.layer_id, expert_idx) for expert_idx in unselected_experts])

        self.experts.prefetch_uid = None
        if hidden_states.shape[0]==1:
            # predict the next layers' experts
            for i in range(1,2):
                layer_index = self.layer_id + i
                topk = 10
                if layer_index >= len(self.layers):
                    break
                next_layer = self.layers[layer_index].mlp
                next_gate = next_layer.gate
                next_routing_logits = next_gate(hidden_states)
                next_routing_weights = F.softmax(next_routing_logits, dim=1, dtype=torch.float)
                next_routing_weights, next_selected_experts = torch.topk(next_routing_weights, topk, dim=-1)
                # next_routing_weights /= next_routing_weights.sum(dim=-1, keepdim=True)
                # next_valid_mask,_,_,_,_ = self.compute_dynamic_expert_routing(next_routing_weights, next_selected_experts, T1=0.3, T2=0.9)
                # next_selected_experts = next_selected_experts[next_valid_mask].tolist()
                next_selected_experts = next_selected_experts[0].tolist()
                uid = self.experts.check(layer_index, next_selected_experts)
                if uid is not None:
                    self.experts.prefetch_uid = uid
                    break
                
        if self.experts.prefetching:
            self.experts.prefetch_lock.wait()
            self.experts.prefetching = False


        # Loop over all available experts in the model and perform the computation on each expert
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True ,high_prec_experts = high_prec_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, high_precision=False) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
        #         *((self.layer_id, expert_idx) for expert_idx in active_experts), 
        #         unordered=True, 
        #         high_prec_experts=high_prec_experts):
            
        #     # idx: 该 Token 在 routing_weights 的第几列 (top-k 中的哪一个)
        #     # top_x: 该 Token 在序列中的全局索引
        #     idx, top_x = torch.where(expert_mask[expert_idx])
        #     assert top_x.shape[0] > 0

        #     # --- 新增逻辑：根据 high_precision_mask 获取该专家下每个 Token 的精度需求 ---
        #     # high_precision_mask 形状与 routing_weights 一致 [num_tokens, top_k]
        #     # 这里取出的 hp_flags 长度与 top_x 一致
        #     hp_flags = high_precision_mask[top_x, idx] 

        #     # 分别获取高精度和低精度 Token 在当前专家输入中的局部索引
        #     hp_indices = torch.where(hp_flags)[0]
        #     lp_indices = torch.where(~hp_flags)[0]

        #     # 遍历两种精度模式进行计算
        #     for is_high_prec, subset_indices in [(True, hp_indices), (False, lp_indices)]:
        #         if subset_indices.shape[0] == 0:
        #             continue  # 如果该专家下没有对应精度的 Token，跳过

        #         # 提取子集对应的全局索引
        #         sub_top_x = top_x[subset_indices]
        #         sub_idx = idx[subset_indices]
                
        #         # 1. 提取对应的隐藏状态
        #         # 优化：直接使用 tensor 索引，避免反复转 list (在 subset 较小时 list 快，较大时 tensor 快)
        #         sub_current_state = hidden_states[sub_top_x] 

        #         # 2. 调用专家层，传入精度标志
        #         # 按照你的要求：通过传入 bool 控制精度
        #         sub_expert_output = expert_layer(sub_current_state, high_precision=is_high_prec)

        #         # 3. 乘上对应的路由权重
        #         # routing_weights 形状通常为 [num_tokens, top_k]
        #         sub_weights = routing_weights[sub_top_x, sub_idx, None]
        #         sub_hidden_states = sub_expert_output * sub_weights

        #         # 4. 累加回全局结果
        #         final_hidden_states.index_add_(
        #             0, 
        #             sub_top_x, 
        #             sub_hidden_states.to(final_hidden_states.dtype)
        #         )
        # if self.experts.prefetch_uid is not None:
        #     self.experts.prefetching=True
        #     self.experts.prefetch(self.experts.registered_experts[self.experts.prefetch_uid])
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            
        return final_hidden_states, router_logits
    
    def prefill_layer_forward(self, layer_id, hidden_states, routing_weights, selected_experts):
        """
        基于双 128 专家缓冲区的 Prefill 阶段前向传播
        """
        # =====================================================================
        # 1. 静态路由：偶数层去 Buffer 0，奇数层去 Buffer 1
        # =====================================================================
        current_buffer = self.buffer_0 if layer_id % 2 == 0 else self.buffer_1
        next_buffer = self.buffer_1 if layer_id % 2 == 0 else self.buffer_0
        
        # =====================================================================
        # 2. 预取收尾与打断（确保当前层数据 100% 就绪）
        # =====================================================================
        # 立即拉起打断信号，要求 PCIe 停止对 next_buffer 的闲杂预取
        self.interrupt_prefetch_signal = True 
        
        # 理论上在 Prefill 阶段，上一层的计算时间足够把这层的 64 个专家全拉过来
        # 但为了极度安全，这里做一个兜底检查：
        active_experts = selected_experts.unique().tolist()
        missing_experts = self.check_missing_in_buffer(current_buffer, active_experts)
        
        if missing_experts:
            # 如果真的没拉完（比如第一层冷启动，或者 PCIe 出现抖动）
            # 直接走高优先级 Demand Stream 阻塞式拉取
            self.demand_load_experts(layer_id, missing_experts, current_buffer)
            
        # 强同步，确保 current_buffer 里当前层需要的专家已经全部在 GPU 上
        self.demand_stream.synchronize()

        # =====================================================================
        # 3. 释放算力：全并行 Batched GEMM 计算
        # (此时使用咱们上一轮写的 parallel_expert_compute)
        # =====================================================================
        final_hidden_states = self.parallel_expert_compute(
            hidden_states, routing_weights, selected_experts, current_buffer
        )

        # =====================================================================
        # 4. 释放总线：启动下一层的全量预取
        # =====================================================================
        # 计算完当前层，解除 PCIe 封锁
        self.interrupt_prefetch_signal = False
        
        if layer_id + 1 < self.total_layers:
            # 开启后台低优先级流 (Prefetch Stream)
            # 因为 next_buffer 容量高达 128，而 Qwen3-VL 只有 64 个专家
            # 我们直接无脑下发 0~63 号专家的拉取指令，不用考虑空间溢出！
            self.start_background_prefetch_all(layer_id + 1, next_buffer)
            
        return final_hidden_states
    
    def parallel_expert_compute(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, 
                                selected_experts: torch.Tensor, current_buffer):
        """
        利用 Batched GEMM 并行计算所有专家 (专供 Prefill 阶段)
        """
        N, hidden_dim = hidden_states.shape
        top_k = selected_experts.shape[-1]
        num_experts = 64 # Qwen3-VL 的满血专家数
        
        # 1. 展平 Token 和专家映射
        flat_hidden_states = hidden_states.repeat_interleave(top_k, dim=0) # [N * top_k, H]
        flat_experts = selected_experts.flatten() # [N * top_k]
        flat_weights = routing_weights.flatten()  # [N * top_k]
        
        # 2. 核心操作：对 Token 按照它要去的专家 ID 进行排序
        sorted_expert_indices = torch.argsort(flat_experts)
        sorted_tokens = flat_hidden_states[sorted_expert_indices]
        sorted_weights = flat_weights[sorted_expert_indices]
        sorted_expert_ids = flat_experts[sorted_expert_indices]
        
        # 3. 统计每个专家名下分到了多少个 Token
        # bincount 可以瞬间统计出 0-63 号专家各自的 Token 数量
        expert_token_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
        
        # 为了使用 torch.bmm，我们需要将长度不一的 Token 组进行 Padding 对齐
        # 找到负载最重的那个专家的 Token 数量
        max_tokens_per_expert = expert_token_counts.max().item()
        
        if max_tokens_per_expert == 0:
            return torch.zeros_like(hidden_states)

        # 4. 构建 Padded 张量 [num_experts, max_tokens, hidden_dim]
        # 初始化为 0，防止 padding 区域产生 NaN
        padded_tokens = torch.zeros(
            (num_experts, max_tokens_per_expert, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # 巧妙地将排序好的连续 Token 填入 Padded 张量
        current_idx = 0
        for i, count in enumerate(expert_token_counts.tolist()):
            if count > 0:
                padded_tokens[i, :count, :] = sorted_tokens[current_idx : current_idx + count]
                current_idx += count

        # =====================================================================
        # 🚀 5. 见证奇迹的时刻：极其暴力的单次并行计算
        # 假设 current_buffer 提供了一次性获取所有专家堆叠权重的接口
        # W1_stacked shape: [num_experts, hidden_dim, intermediate_dim]
        # =====================================================================
        W1_stacked, W2_stacked = current_buffer.get_stacked_weights()
        
        # [num_experts, max_tokens, intermediate_dim]
        intermediate_states = torch.bmm(padded_tokens, W1_stacked)
        intermediate_states = torch.nn.functional.silu(intermediate_states) # 激活函数
        
        # [num_experts, max_tokens, hidden_dim]
        output_padded = torch.bmm(intermediate_states, W2_stacked)
        # =====================================================================

        # 6. 把计算结果从 Padded 张量中按原顺序抽出来
        sorted_outputs = torch.empty_like(sorted_tokens)
        current_idx = 0
        for i, count in enumerate(expert_token_counts.tolist()):
            if count > 0:
                sorted_outputs[current_idx : current_idx + count] = output_padded[i, :count, :]
                current_idx += count
                
        # 7. 乘上路由权重
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(-1)
        
        # 8. 乱序还原并累加回最终结果
        # 因为最初做了一次 argsort，我们需要把结果 scatter 回原始的 Token 位置
        final_hidden_states = torch.zeros_like(hidden_states)
        original_indices = sorted_expert_indices // top_k # 还原出属于哪个原始 Token
        
        final_hidden_states.index_add_(0, original_indices, sorted_outputs)
        
        return final_hidden_states
    
        
    def compute_dynamic_expert_routing(self, routing_weights: torch.Tensor, selected_experts: torch.Tensor, T1: float = 0.6, T2: float = 0.9):
        """
        动态评估专家的精度需求并过滤被跳过的专家。
        返回: 
            valid_mask (Tensor): 布尔掩码，标记哪些 (Token, Expert) 分配是有效的(未被跳过)。
            high_prec_experts (set): 全局需要高精度加载的专家索引集合。
            skipped_ratio (float): 被跳过的路由比例。
            low_prec_ratio (float): 使用低精度的路由比例。
        """
        # routing_weights 形状: (N, top_k), 且已经被 torch.topk 按从大到小排序
        
        # 1. 计算【在加入当前专家前】的累计置信度得分
        cumsum_weights = torch.cumsum(routing_weights, dim=-1)
        # 将累计值向右平移，因为当前专家的定级取决于排在它前面的专家得分之和
        cumsum_before = cumsum_weights - routing_weights
        
        # 2. 生成掩码 (Masks)
        # 累计得分 < T1 的标记为高精度
        high_precision_mask = torch.full_like(cumsum_before, False, dtype=torch.bool)
        # if routing_weights.shape[0] > 1:
        #     high_precision_mask[~self.token_type] = cumsum_before[~self.token_type] <= T1
        #     high_precision_mask[self.token_type] = cumsum_before[self.token_type] <= 0.6
        # else:
        #     high_precision_mask = cumsum_before <= 0.6
        high_precision_mask = cumsum_before <= 0.0
        # T1 <= 累计得分 < T2 的标记为低精度
        # low_precision_mask = (cumsum_before >= T1) & (cumsum_before < T2)
        # 累计得分 >= T2 的标记为跳过 (Skip)
        skip_mask = cumsum_before >= T2
        
        # 有效掩码：只要没被跳过，就是有效的
        valid_mask = ~skip_mask
        # skipped_experts = set(selected_experts[skip_mask].unique().tolist())
        # 3. 统计比例 (用于毕设指标记录)
        # total_routing_edges = routing_weights.numel() # N * top_k
        # # skipped_ratio = skip_mask.sum().item() / total_routing_edges
        # origin_experts = selected_experts.unique().tolist()
        # 4. 提取出需要高精度的专家列表
        # 只要该专家在全批次中被任何一个 Token 标记为高精度，物理加载时就必须走高精度
        high_prec_experts = set(selected_experts[high_precision_mask].unique().tolist())
        # low_prec_experts = set(selected_experts[~high_precision_mask].unique().tolist())
        # # low_prec_ratio = (len(origin_experts) - len(high_prec_experts)) / len(origin_experts) if origin_experts else 0
        # # skipped_ratio = len(skipped_experts) / len(origin_experts) if origin_experts else 0
        # hobb_load = len(high_prec_experts) + len(low_prec_experts)*0.5
        # base_load = len(origin_experts)
        # my_load = len(high_prec_experts) + (len(origin_experts) - len(high_prec_experts))*0.5
        hobb_load = 0
        base_load = 0
        my_load = 0
        # return valid_mask, high_prec_experts, skipped_ratio, low_prec_ratio
        return valid_mask, high_prec_experts, hobb_load, base_load , high_precision_mask



class QwenMoeBlock_NoQuant(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建三个临时的 Linear 层，后续会被 Wrapper 的 storage 覆盖
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)