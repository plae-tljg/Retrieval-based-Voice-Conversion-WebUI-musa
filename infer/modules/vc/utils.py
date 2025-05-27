import os

from fairseq import checkpoint_utils

import torch
import torch.nn.functional as F
from torch import nn


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def musa_attention_wrapper(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Wrapper for scaled dot product attention to handle MUSA GPU compatibility
    
    Args:
        q: Query tensor
        k: Key tensor 
        v: Value tensor
        attn_mask: Attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        
    Returns:
        attn_output: Attention output
    """
    # 直接使用手动实现，避免任何递归调用
    scaling = float(q.size(-1)) ** -0.5
    q = q * scaling
    
    # 计算注意力分数
    attn = torch.matmul(q, k.transpose(-2, -1))
    
    # 处理 mask
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            bsz, num_heads, tgt_len, src_len = attn_mask.shape
            attn_mask = attn_mask.view(bsz * num_heads, tgt_len, src_len)
        if attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
        attn = attn + attn_mask
    
    # 应用 softmax
    attn = F.softmax(attn, dim=-1)
    
    # 应用 dropout
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    # 计算输出
    output = torch.matmul(attn, v)
    
    return output

# 替换原始的 scaled_dot_product_attention
F.scaled_dot_product_attention = musa_attention_wrapper
