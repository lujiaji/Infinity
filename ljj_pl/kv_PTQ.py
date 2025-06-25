import time
import torch
import numpy as np
from ljj_pl.Q_tools import PTQ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KV_PTQ:
    """Key-Value Post Training Quantization 实现
    
    该类用于对Transformer模型中的Key和Value矩阵进行量化。支持多种量化模式：
    - per-token: 按token维度量化
    - per-head: 按注意力头量化
    - per-head+per-dim: 按注意力头和维度联合量化
    - per-head+per-scale: 按注意力头和scale联合量化
    - per-head+per-dim+per-scale: 三者联合量化
    - kv_diff_method: K和V使用不同的量化方法
    - kv_diff_method+per-batch: 在不同方法基础上加入batch维度
    - K-only/V-only: 仅量化K或V
    - per-batch: 按batch维度量化
    - per-tensor: 整个张量统一量化
    
    Attributes:
        new_k: 当前需要量化的K矩阵
        new_v: 当前需要量化的V矩阵
        cached_k: 已量化的K矩阵缓存
        cached_v: 已量化的V矩阵缓存
        cached_s_k: K矩阵的量化scale缓存
        cached_s_v: V矩阵的量化scale缓存
        gen_k: 累积的K矩阵（用于特定模式）
        gen_v: 累积的V矩阵（用于特定模式）
        q_bits: 量化位数
        dim_cat: 拼接维度
        q_dim: 量化维度模式
        block_idx: 块索引
        use_diff_bits: 是否在特定情况下使用不同的量化位数
    """
    
    def __init__(self, k: torch.Tensor, v: torch.Tensor, q_bits: int, 
                 dim_cat: int, q_dim: str, blk_idx: int, use_diff_bits: bool = False):
        """初始化KV_PTQ对象
        
        Args:
            k: Key矩阵，形状为[batch_size, seq_len, num_heads, head_dim]
            v: Value矩阵，形状为[batch_size, seq_len, num_heads, head_dim]
            q_bits: 量化位数，通常为4或8
            dim_cat: 在哪个维度上进行拼接，用于处理多个batch的情况
            q_dim: 量化维度的模式，可选值参见类文档
            blk_idx: Transformer块的索引，用于区分不同层
            use_diff_bits: 是否在序列长度较大时使用不同的量化位数，默认为False
        """
        self.new_k, self.new_v = k, v
        self.cached_k, self.cached_v = None, None
        self.q_bits, self.dim_cat, self.q_dim, self.block_idx = q_bits, dim_cat, q_dim, blk_idx
        self.cached_s_k, self.cached_s_v = None, None
        self.gen_k, self.gen_v = None, None
        self.mark = False
        self.use_diff_bits = use_diff_bits
        self.seq_len = []

    def quantized_kv(self):
        """执行K和V矩阵的量化操作
        
        根据不同的量化模式(q_dim)选择相应的量化策略：
        1. 对于常规模式：直接量化并缓存结果
        2. 对于累积模式(per-head+per-dim等)：先累积再量化
        3. 对于特殊模式(K-only等)：使用整体量化
        
        特别地，当序列长度>=100时，使用4bit量化，否则使用默认位数
        """
        def init_PTQ(q_bits):
            if self.q_dim=='kv_diff_method':
                do_k=PTQ(ckpt=self.new_k,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim='per-head+per-scale')
                do_v=PTQ(ckpt=self.new_v,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim='per-token')
                
            elif self.q_dim=='kv_diff_method+per-batch':
                do_k=PTQ(ckpt=self.new_k,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim='per-head+per-scale+per-batch')
                do_v=PTQ(ckpt=self.new_v,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim='per-token+per-batch')

            # elif self.q_dim in ('per-head+per-dim','per-head'):
            #     do_k=PTQ(ckpt=self.gen_k,
            #         q_mode='symmetric',
            #         q_bits=self.q_bits,
            #         group='per_channel',
            #         target='single',
            #         kv=True,
            #         q_dim=self.q_dim)
            #     do_v=PTQ(ckpt=self.gen_v,
            #         q_mode='symmetric',
            #         q_bits=self.q_bits,
            #         group='per_channel',
            #         target='single',
            #         kv=True,
            #         q_dim=self.q_dim)
                
            elif self.q_dim in ('K-only','V-only','per-tensor'):
                do_k=PTQ(ckpt=self.new_k,
                    q_mode='symmetric',
                    q_bits=q_bits if q_bits else self.q_bits,
                    group='whole',
                    target='single',
                    kv=True,
                    q_dim=self.q_dim)
                do_v=PTQ(ckpt=self.new_v,
                    q_mode='symmetric',
                    q_bits=q_bits if q_bits else self.q_bits,
                    group='whole',
                    target='single',
                    kv=True,
                    q_dim=self.q_dim)
                
            else:
                do_k=PTQ(ckpt=self.new_k,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim=self.q_dim)
                do_v=PTQ(ckpt=self.new_v,
                    q_mode='symmetric',
                    q_bits=self.q_bits,
                    group='per_channel',
                    target='single',
                    kv=True,
                    q_dim=self.q_dim)
                
            return do_k,do_v

        if self.q_dim in ('per-token','per-head+per-scale','per-head+per-dim+per-scale',
                          'kv_diff_method','kv_diff_method+per-batch','K-only','V-only',
                          'per-batch','per-tensor'):
            if self.use_diff_bits and self.new_k.shape[1]>=100:
                do_k,do_v=init_PTQ(4)
            else:
                do_k,do_v=init_PTQ(None)
            do_k.quantize();do_v.quantize()
            if self.q_dim=='per-batch':
                q_k,q_v=do_k.quantized_item,do_v.quantized_item
                s_k,s_v=do_k.scale.expand(-1,q_k.shape[1],-1,-1),do_v.scale.expand(-1,q_k.shape[1],-1,-1)
            else:
                q_k,q_v,s_k,s_v=do_k.quantized_item,do_v.quantized_item,do_k.scale,do_v.scale
            del do_k,do_v
            if self.cached_k==None:
                self.cached_k,self.cached_v,self.cached_s_k,self.cached_s_v=q_k,q_v,s_k,s_v
            else:
                self.cached_k=torch.cat((self.cached_k, q_k), dim=self.dim_cat)
                self.cached_v=torch.cat((self.cached_v, q_v), dim=self.dim_cat)
                self.cached_s_k=torch.cat((self.cached_s_k, s_k), dim=self.dim_cat)
                self.cached_s_v=torch.cat((self.cached_s_v, s_v), dim=self.dim_cat)

        elif self.q_dim in ('per-head+per-dim','per-head'):
            if self.use_diff_bits and self.block_idx<12:
                do_k,do_v=init_PTQ(4)
            else:
                do_k,do_v=init_PTQ(None)
            self.seq_len.append(self.new_k.shape[self.dim_cat])
            do_k.quantize();do_v.quantize()
            q_k,q_v,s_k,s_v=do_k.quantized_item.contiguous(),do_v.quantized_item.contiguous(),do_k.scale.contiguous(),do_v.scale.contiguous()
            del do_k,do_v
            if self.cached_s_k==None:
                self.cached_s_k,self.cached_s_v=s_k,s_v
            else:
                self.cached_s_k=torch.cat((self.cached_s_k, s_k), dim=self.dim_cat)
                self.cached_s_v=torch.cat((self.cached_s_v, s_v), dim=self.dim_cat)
            if self.cached_k==None:
                self.cached_k,self.cached_v=q_k,q_v
            else:
                self.cached_k=torch.cat((self.cached_k, q_k), dim=self.dim_cat)
                self.cached_v=torch.cat((self.cached_v, q_v), dim=self.dim_cat)

    def use_kv(self):
        """使用量化后的K和V矩阵
        
        Returns:
            tuple: (cur_k, cur_v)
                - cur_k: 反量化后的K矩阵
                - cur_v: 反量化后的V矩阵
        """
        if self.q_dim in ('per-head+per-dim','per-head'):
            assert len(self.seq_len)==self.cached_s_k.shape[2],"seq_len and cached_s_k.shape[self.dim_cat] must be the same"
            cur_s_k,cur_s_v=None,None
            for idx,len_seq in enumerate(self.seq_len):
                if cur_s_k==None:
                    cur_s_k=self.cached_s_k[:,:,0:1,:].expand(-1,-1,len_seq,-1).contiguous()
                    cur_s_v=self.cached_s_v[:,:,0:1,:].expand(-1,-1,len_seq,-1).contiguous()
                else:
                    cur_s_k=torch.cat((cur_s_k,self.cached_s_k[:,:,idx:idx+1,:].expand(-1,-1,len_seq,-1)),dim=self.dim_cat).contiguous()
                    cur_s_v=torch.cat((cur_s_v,self.cached_s_v[:,:,idx:idx+1,:].expand(-1,-1,len_seq,-1)),dim=self.dim_cat).contiguous()
            cur_k = (self.cached_k.float() * cur_s_k).to(torch.float32)
            cur_v = (self.cached_v.float() * cur_s_v).to(torch.float32)
            del cur_s_k,cur_s_v
        else:
            cur_k = (self.cached_k.float() * self.cached_s_k).to(torch.float32)
            cur_v = (self.cached_v.float() * self.cached_s_v).to(torch.float32)
        return cur_k, cur_v
  
    def record_kv(self,k,v):
        # self.save_k_npz_path=f'/data/jiaji_lu/kv/{self.q_dim}/q_k_{self.q_dim}_{self.block_idx}.npz'
        # self.save_v_npz_path=f'/data/jiaji_lu/kv/{self.q_dim}/q_v_{self.q_dim}_{self.block_idx}.npz'
        self.save_k_npz_path=f'/data/jiaji_lu/kv/ori/k_ori_{self.block_idx}.npz'
        self.save_v_npz_path=f'/data/jiaji_lu/kv/ori/v_ori_{self.block_idx}.npz'
        np.savez(self.save_k_npz_path,data=k.cpu().numpy())
        np.savez(self.save_v_npz_path,data=v.cpu().numpy())
        print(f'{self.block_idx} done save kv!')

    def record_q(self,q):
        # self.save_q_npz_path=f'/data/jiaji_lu/kv/q/q_ori_{self.block_idx}'
        self.save_q_npz_path=f'/data/jiaji_lu/kv/q/q_q_{self.q_dim}_{self.block_idx}.npz'
        np.savez(self.save_q_npz_path,data=q.cpu().numpy())
        print(f'{self.block_idx} done save q!')