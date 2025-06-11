from typing import Dict, Union, Optional, Tuple
import torch
from torch import Tensor

class PTQ():
    """Post Training Quantization 实现
    
    支持多种量化模式：
    - whole: 整体量化
    - per_channel: 按通道量化
    - per_block: 按块量化
    """
    VALID_GROUPS = {'per_channel', 'per_block', 'whole'}
    VALID_TARGETS = {'whole', 'single'}
    VALID_Q_DIMS = {'per-token', 'per-head+per-scale', 'per-head+per-dim+per-scale', 
                    'per-head+per-dim', 'per-head', 'per-token+per-batch', 
                    'per-head+per-scale+per-batch', 'per-batch', 'K-only', 'V-only', 'per-tensor'}
    TARGET_SUBSTRINGS = ["attn.mat_qkv.weight", "attn.proj.weight", 
                        "ffn.fc1.weight", "ffn.fc2.weight", "ada_lin.1.weight"]

    def __init__(self, 
                 ckpt: Union[str, Dict[str, Tensor]], 
                 q_mode: str = 'symmetric',
                 q_bits: int = 8,
                 group: str = 'per_channel',
                 group_sz: int = 16,
                 target: str = 'whole',
                 kv: bool = False,
                 q_dim: str = 'token') -> None:
        """
        初始化PTQ（Post-Training Quantization）对象
        
        Args:
            ckpt: 模型检查点或权重字典
            q_mode: 量化模式，默认为'symmetric'
            q_bits: 量化位数，默认为8
            group: 分组方式，可选'per_channel'/'per_block'/'whole'
            group_sz: 分组大小，当group='per_block'时使用
            target: 目标类型，可选'whole'/'single'
            kv: 是否为KV量化
            q_dim: 量化维度
        """
        import warnings
        if group not in self.VALID_GROUPS:
            warnings.warn(f"Unexpected group: {group}")
        if target not in self.VALID_TARGETS:
            warnings.warn(f"Unexpected target: {target}")
        if q_dim not in self.VALID_Q_DIMS:
            raise ValueError(f"q_dim must be one of {self.VALID_Q_DIMS}")

        self.ckpt = ckpt
        self.q_mode = q_mode
        self.group = group
        self.target = target
        self.kv = kv
        self.q_dim = q_dim
        self.group_sz = group_sz if group == 'per_block' else 1
        
        self.quantized_item = {}
        self.scale = {}
        self.q_bits = q_bits
        
        # 特殊处理1bit量化
        if q_bits == 1:
            self.bound_min = 0
            self.bound_max = 1
            self.is_1bit = True
        else:
            self.bound_min = -(2**(q_bits-1))
            self.bound_max = 2**(q_bits-1)-1
            self.is_1bit = False

    def _compute_scale(self, max_val: Tensor) -> Tensor:
        if self.is_1bit:
            # 对于1bit量化，直接返回最大值作为scale
            return max_val
        else:
            return max_val / self.bound_max

    def _quantize_block(self, block: Tensor) -> Tuple[Tensor, Tensor]:
        max_val = block.abs().amax()
        scale = self._compute_scale(max_val)
        
        if self.is_1bit:
            # 1bit量化：大于0为1，小于等于0为0
            quantized_block = (block > 0).to(torch.int8)
        else:
            quantized_block = torch.round(block/scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
        return quantized_block, scale

    def _quantize_whole(self) -> None:
        self.w_all = torch.load(self.ckpt) if isinstance(self.ckpt, str) else self.ckpt
        
        for key in self.w_all:
            if not (key.startswith("blocks.") and any(sub in key for sub in self.TARGET_SUBSTRINGS)):
                continue

            if self.group == 'whole':
                max_val = self.w_all[key].abs().amax()
                scale = self._compute_scale(max_val)
                
                if self.is_1bit:
                    self.quantized_item[key] = (self.w_all[key] > 0).to(torch.int8)
                else:
                    self.quantized_item[key] = torch.round(self.w_all[key]/scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
                self.scale[key] = scale.expand_as(self.w_all[key].shape[0], 1)

            elif self.group == 'per_channel':
                max_val = self.w_all[key].abs().amax(dim=0, keepdim=True)
                scale = self._compute_scale(max_val)
                
                if self.is_1bit:
                    self.quantized_item[key] = (self.w_all[key] > 0).to(torch.int8)
                else:
                    self.quantized_item[key] = torch.round(self.w_all[key]/scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
                self.scale[key] = scale

            elif self.group == 'per_block':
                quantized_blocks, scale_blocks = [], []
                num_block = self.w_all[key].shape[0] // self.group_sz
                remainder = self.w_all[key].shape[0] % self.group_sz
                
                for n in range(num_block):
                    block = self.w_all[key][self.group_sz*n : self.group_sz*(n+1)]
                    quantized_block, scale = self._quantize_block(block)
                    quantized_blocks.append(quantized_block)
                    scale_blocks.append(scale.expand_as(block))
                
                if remainder > 0:
                    block = self.w_all[key][num_block*self.group_sz:]
                    quantized_block, scale = self._quantize_block(block)
                    quantized_blocks.append(quantized_block)
                    scale_blocks.append(scale.expand_as(block))
                
                self.quantized_item[key] = torch.cat(quantized_blocks, dim=0)
                self.scale[key] = torch.cat(scale_blocks, dim=0)

    def _quantize_single(self) -> None:
        self.w_all = self.ckpt
        
        if self.group == 'whole':
            max_val = self.w_all.abs().amax()
            scale = self._compute_scale(max_val)
            
            if self.is_1bit:
                self.quantized_item = (self.w_all > 0).to(torch.int8)
            else:
                self.quantized_item = torch.round(self.w_all / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
            self.scale = scale.view(1, 1, 1, 1).expand(self.w_all.shape[0], self.w_all.shape[1], 1, 1)
        
        elif self.group == 'per_channel':
            if self.kv and self.q_dim:
                dim_map = {
                    'per-token': (0, 1, 3),
                    'per-head+per-token': (0, 3),
                    'per-head+per-dim+per-scale': (0,),
                    'per-head+per-dim': (0, 2),
                    'per-head': (0, 2, 3),
                    'per-token+per-batch': (2, 3),
                    'per-head+per-scale+per-batch': (3,),
                    'per-batch': (1, 2, 3)
                }
                if self.q_dim not in dim_map and self.q_dim not in {'K-only', 'V-only', 'per-tensor'}:
                    raise ValueError(f"Invalid q_dim '{self.q_dim}' for per_channel quantization with kv=True. Must be one of {list(dim_map.keys())}")
                max_val = self.w_all.abs().amax(dim=dim_map[self.q_dim], keepdim=True)
            else:
                max_val = self.w_all.abs().amax(dim=tuple(range(self.w_all.ndim-1)), keepdim=True)
            
            scale = self._compute_scale(max_val)
            
            if self.is_1bit:
                self.quantized_item = (self.w_all > 0).to(torch.int8)
            else:
                self.quantized_item = torch.round(self.w_all / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
            self.scale = scale
        
        elif self.group == 'per_block':
            quantized_blocks, scale_blocks = [], []
            num_block = self.w_all.shape[0] // self.group_sz
            remainder = self.w_all.shape[0] % self.group_sz
            
            for n in range(num_block):
                block = self.w_all[self.group_sz*n : self.group_sz*(n+1)]
                quantized_block, scale = self._quantize_block(block)
                quantized_blocks.append(quantized_block)
                scale_blocks.append(scale.view(1, 1).expand(block.shape[0], 1))
            
            if remainder > 0:
                block = self.w_all[num_block*self.group_sz:]
                quantized_block, scale = self._quantize_block(block)
                quantized_blocks.append(quantized_block)
                scale_blocks.append(scale.view(1, 1).expand(block.shape[0], 1))
            
            self.quantized_item = torch.cat(quantized_blocks, dim=0)
            self.scale = torch.cat(scale_blocks, dim=0)

    def quantize(self) -> None:
        try:
            if self.target == 'whole':
                self._quantize_whole()
            elif self.target == 'single':
                self._quantize_single()
            else:
                raise ValueError(f"Unsupported target: {self.target}")
        except Exception as e:
            print(f"Quantization failed: {str(e)}")
            raise

    def dequantize(self) -> Union[Dict[str, Tensor], Tensor]:
        if self.target == 'whole':
            for key in self.quantized_item:
                self.w_all[key] = (self.quantized_item[key].float() * self.scale[key]).to(torch.float32)
            return self.w_all
        else:
            return (self.quantized_item.float() * self.scale).to(torch.float32)

    def fake_q_deq(self) -> Union[Dict[str, Tensor], Tensor]:
        self.quantize()
        out = self.dequantize()
        self.cleanup()
        return out

    def cleanup(self) -> None:
        attrs_to_clean = ['w_all', 'quantized_item', 'scale', 'ckpt']
        for attr in attrs_to_clean:
            if hasattr(self, attr):
                delattr(self, attr)

    def verify_quantization(self) -> Optional[Dict[str, float]]:
        try:
            original = self.w_all.clone()
            quantized = self.fake_q_deq()
            error = (original - quantized).abs()
            return {
                'mean_error': error.mean().item(),
                'max_error': error.max().item()
            }
        except Exception:
            return None

    def __del__(self):
        self.cleanup()