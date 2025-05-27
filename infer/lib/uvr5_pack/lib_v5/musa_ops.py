import torch
import torch_musa
import torch.nn.functional as F

def replication_pad2d(input, padding):
    """
    自定义的replication_pad2d实现，用于MUSA后端
    
    Args:
        input (torch.Tensor): 输入张量，形状为(N, C, H, W)
        padding (tuple): 填充大小，格式为(pad_left, pad_right, pad_top, pad_bottom)
    
    Returns:
        torch.Tensor: 填充后的张量
    """
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        # 如果不是MUSA后端，使用PyTorch原生实现
        return F.pad(input, padding, mode='replicate')
    
    # 解析padding
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    # 获取输入尺寸
    N, C, H, W = input.shape
    
    # 创建输出张量
    output = torch.zeros(N, C, H + pad_top + pad_bottom, W + pad_left + pad_right, 
                        device=input.device, dtype=input.dtype)
    
    # 复制原始数据到中心区域
    output[:, :, pad_top:pad_top+H, pad_left:pad_left+W] = input
    
    # 填充顶部
    if pad_top > 0:
        output[:, :, :pad_top, pad_left:pad_left+W] = input[:, :, 0:1, :].expand(-1, -1, pad_top, -1)
    
    # 填充底部
    if pad_bottom > 0:
        output[:, :, pad_top+H:, pad_left:pad_left+W] = input[:, :, -1:, :].expand(-1, -1, pad_bottom, -1)
    
    # 填充左侧
    if pad_left > 0:
        output[:, :, :, :pad_left] = output[:, :, :, pad_left:pad_left+1].expand(-1, -1, -1, pad_left)
    
    # 填充右侧
    if pad_right > 0:
        output[:, :, :, pad_left+W:] = output[:, :, :, pad_left+W-1:pad_left+W].expand(-1, -1, -1, pad_right)
    
    return output

# 替换torch.nn.functional.pad中的replicate模式
def pad(input, padding, mode='constant', value=0):
    """
    重写pad函数，在MUSA后端上使用自定义的replication_pad2d
    """
    if mode == 'replicate' and len(padding) == 4 and input.dim() == 4:
        return replication_pad2d(input, padding)
    return F.pad(input, padding, mode=mode, value=value) 