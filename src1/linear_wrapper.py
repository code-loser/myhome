import torch
from torch import nn
import typing as tp

class QwenLinearWrapper(nn.Module):
    def __init__(self, linear_module: nn.Linear, device: torch.device):
        super().__init__()
        # 直接接管原始 linear 的 weight 和 bias
        self.linear_module, self.storage = self.replace_layer_storage(linear_module, device)
        
    @staticmethod
    def replace_layer_storage(layer: nn.Linear, device: torch.device):
        # 获取原始权重（FP16/BF16）
        params = {"weight": layer.weight}
        if layer.bias is not None:
            params["bias"] = layer.bias

        total_bytes = sum(p.nbytes for p in params.values())
        # 创建一个统一的存储空间
        storage = torch.UntypedStorage(total_bytes, device=device)
        
        offset = 0
        new_params = {}
        for name, p in params.items():
            # 在 storage 上创建视图并复制数据
            p_view = torch.as_tensor(storage[offset : offset + p.nbytes], dtype=p.dtype, device=device).view(p.shape)
            p_view.copy_(p)
            new_params[name] = nn.Parameter(p_view)
            offset += p.nbytes
            
        # 更新原始层，使其指向新的 storage 视图
        layer.weight = new_params["weight"]
        if layer.bias is not None:
            layer.bias = new_params["bias"]
            
        return layer, storage

    def forward(self, x):
        return self.linear_module(x)
