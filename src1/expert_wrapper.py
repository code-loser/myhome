import typing as tp
import torch
from torch import nn
from .utils import nested_flatten, nested_pack
from .linear_wrapper import QwenLinearWrapper
from transformers.activations import ACT2FN
import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenExpertWrapper(nn.Module):
    def __init__(self, shape_gate_up, shape_down, device):
        super().__init__()
        # 仅仅保留 4 个占位符，不放真实数据
        self.register_buffer('gate_up_base', torch.empty(shape_gate_up, dtype=torch.uint8, device=device))
        self.register_buffer('gate_up_residual', torch.empty(shape_gate_up, dtype=torch.uint8, device=device))
        self.register_buffer('down_base', torch.empty(shape_down, dtype=torch.uint8, device=device))
        self.register_buffer('down_residual', torch.empty(shape_down, dtype=torch.uint8, device=device))

    def point_to(self, 
                 target_gu_base: torch.UntypedStorage, 
                 target_gu_res: torch.UntypedStorage, 
                 target_d_base: torch.UntypedStorage, 
                 target_d_res: torch.UntypedStorage):
        """
        核心魔法：零拷贝显存重定向 (直接接收底层 Storage)
        """
        # 参数释义：set_(底层显存对象, 起始偏移量, 形状, 步长)
        # 因为传进来的直接就是 Storage，我们偏移量设为 0，形状和步长直接用自己的！
        self.gate_up_base.set_(
            target_gu_base, 0, self.gate_up_base.size(), self.gate_up_base.stride()
        )
        self.gate_up_residual.set_(
            target_gu_res, 0, self.gate_up_residual.size(), self.gate_up_residual.stride()
        )
        
        self.down_base.set_(
            target_d_base, 0, self.down_base.size(), self.down_base.stride()
        )
        self.down_residual.set_(
            target_d_res, 0, self.down_residual.size(), self.down_residual.stride()
        )

    def forward(self, x ,high_precision = False):
        # 🌟 无分支的极致执行流 🌟
        # 无论传进来的是真实残差，还是全局全零矩阵，直接算！
        if high_precision:
            gate_up_bf16 = reconstruct_to_bf16_triton(self.gate_up_base, self.gate_up_residual ,self.gate_up_base.device)
            down_bf16 = reconstruct_to_bf16_triton(self.down_base, self.down_residual, self.down_base.device)
        else:
            temp_tensor = torch.full_like(self.gate_up_residual, 240, dtype=torch.uint8, device=self.gate_up_base.device)
            gate_up_bf16 = reconstruct_to_bf16_triton(self.gate_up_base, temp_tensor ,self.gate_up_base.device)
            down_bf16 = reconstruct_to_bf16_triton(self.down_base, temp_tensor, self.down_base.device)
        gate_up = x @ gate_up_bf16 
        
        # 2. 切分与激活
        gate, up = gate_up.chunk(2, dim=-1)   
        activated = F.silu(gate) * up
        
        # 3. 降维输出，同样直接相乘
        # activated: (batch, 768), down_bf16: (768, 2048) -> 输出 (batch, 2048)
        return activated @ down_bf16


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def _reconstruct_bf16_kernel_unified(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    a_i32 = a.to(tl.int32)
    b_i32 = b.to(tl.int32)

    S = (a_i32 & 0x80) << 8
    M_high = (a_i32 & 0x07) << 4
    M_low = b_i32 & 0x0F
    
    E_mod = (a_i32 >> 3) & 0x0F
    b_high = (b_i32 >> 4) & 0x0F
    
    E_val = E_mod + 100 + b_high
    E = E_val << 7

    # zero_mask = (a_i32 == 0) & (b_i32 == 0)
    # E = tl.where(zero_mask, 0, E)

    c_i32 = S | E | M_high | M_low
    c_i16 = c_i32.to(tl.int16)

    tl.store(c_ptr + offsets, c_i16, mask=mask)

def reconstruct_to_bf16_triton(a: torch.Tensor, b: torch.Tensor ,target_device: torch.device) -> torch.Tensor:
    if not a.is_cuda: a = a.to(target_device)
    if not b.is_cuda: b = b.to(target_device)
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    c_i16 = torch.empty(n_elements, device=a.device, dtype=torch.int16)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch.cuda.device(target_device):
        _reconstruct_bf16_kernel_unified[grid](a, b, c_i16, n_elements)
    return c_i16.view(torch.bfloat16).view(a.shape)