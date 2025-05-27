import torch
import torch_musa
import torch.nn as nn
import torch.nn.functional as F

def _compute_norm(v, dim, keepdim=True):
    """
    使用基础操作计算范数，避免使用 torch.norm
    Args:
        v: 输入张量
        dim: 计算范数的维度
        keepdim: 是否保持维度
    Returns:
        计算得到的范数
    """
    # 使用平方和开根号的方式计算范数
    v_square = torch.clamp(v * v, min=0.0, max=1e6)  # 限制平方值的范围
    v_sum = torch.sum(v_square, dim=dim, keepdim=keepdim)
    v_norm = torch.sqrt(torch.clamp(v_sum + 1e-8, min=1e-8, max=1e6))  # 限制范数的范围
    return v_norm

def _weight_norm_interface_forward(v, g, dim):
    """
    实现weight normalization的前向传播
    Args:
        v: 权重向量
        g: 缩放因子
        dim: 归一化维度
    Returns:
        归一化后的权重
    """
    # 限制输入值的范围
    v = torch.clamp(v, min=-1e6, max=1e6)
    g = torch.clamp(g, min=-1e6, max=1e6)
    
    # 使用自定义的范数计算函数
    v_norm = _compute_norm(v, dim=dim, keepdim=True)
    # 归一化v
    v_normalized = v / (v_norm + 1e-8)
    # 应用缩放因子
    return torch.clamp(v_normalized * g, min=-1e6, max=1e6)

def _weight_norm_interface_backward(grad_output, v, g, dim):
    """
    实现weight normalization的反向传播
    Args:
        grad_output: 输出梯度
        v: 权重向量
        g: 缩放因子
        dim: 归一化维度
    Returns:
        v的梯度和g的梯度
    """
    # 限制输入值的范围
    v = torch.clamp(v, min=-1e6, max=1e6)
    g = torch.clamp(g, min=-1e6, max=1e6)
    grad_output = torch.clamp(grad_output, min=-1e6, max=1e6)
    
    # 使用自定义的范数计算函数
    v_norm = _compute_norm(v, dim=dim, keepdim=True)
    v_normalized = v / (v_norm + 1e-8)
    
    # 计算g的梯度
    grad_g = torch.sum(grad_output * v_normalized, dim=dim, keepdim=True)
    grad_g = torch.clamp(grad_g, min=-1e6, max=1e6)
    
    # 计算v的梯度
    grad_v = grad_output * g
    grad_v = grad_v - v_normalized * torch.sum(grad_v * v_normalized, dim=dim, keepdim=True)
    grad_v = grad_v / (v_norm + 1e-8)
    grad_v = torch.clamp(grad_v, min=-1e6, max=1e6)
    
    return grad_v, grad_g

class WeightNormMUSA(nn.Module):
    """
    MUSA版本的WeightNorm实现
    """
    def __init__(self, module, name='weight', dim=0):
        super(WeightNormMUSA, self).__init__()
        self.module = module
        self.name = name
        self.dim = dim
        
        # 获取原始权重
        w = getattr(self.module, self.name)
        del self.module._parameters[self.name]
        
        # 初始化v和g
        self.module.register_parameter(self.name + '_v', nn.Parameter(w.data))
        self.module.register_parameter(self.name + '_g', nn.Parameter(torch.norm(w.data, dim=self.dim)))
        
    def forward(self, *args, **kwargs):
        # 计算归一化后的权重
        v = getattr(self.module, self.name + '_v')
        g = getattr(self.module, self.name + '_g')
        w = _weight_norm_interface_forward(v, g, self.dim)
        
        # 设置权重
        setattr(self.module, self.name, w)
        
        # 调用原始模块的前向传播
        return self.module(*args, **kwargs)
    
    def remove(self):
        """
        移除weight normalization,恢复原始权重
        """
        v = getattr(self.module, self.name + '_v')
        g = getattr(self.module, self.name + '_g')
        w = _weight_norm_interface_forward(v, g, self.dim)
        
        del self.module._parameters[self.name + '_v']
        del self.module._parameters[self.name + '_g']
        self.module.register_parameter(self.name, nn.Parameter(w.data))

def weight_norm_musa(module, name='weight', dim=0):
    """
    应用weight normalization到模块
    Args:
        module: 要应用weight norm的模块
        name: 权重参数名
        dim: 归一化维度
    Returns:
        包装后的模块
    """
    return WeightNormMUSA(module, name, dim)

def remove_weight_norm(module, name='weight'):
    """
    从模块中移除weight normalization
    Args:
        module: 要移除weight norm的模块
        name: 权重参数名
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNormMUSA) and hook.name == name:
            hook.remove()
            del module._forward_pre_hooks[k]
            return module
    raise ValueError(f"weight_norm of name '{name}' not found in {module}")

class WeightNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        ctx.dim = dim
        # 限制输入值的范围
        v = torch.clamp(v, min=-1e6, max=1e6)
        g = torch.clamp(g, min=-1e6, max=1e6)
        
        v_norm = _compute_norm(v, dim=dim, keepdim=True)
        v_normalized = v / (v_norm + 1e-8)
        ctx.save_for_backward(v, g, v_normalized, v_norm)
        return torch.clamp(v_normalized * g, min=-1e6, max=1e6)

    @staticmethod
    def backward(ctx, grad_output):
        v, g, v_normalized, v_norm = ctx.saved_tensors
        dim = ctx.dim
        
        # 限制梯度范围
        grad_output = torch.clamp(grad_output, min=-1e6, max=1e6)
        
        # 计算g的梯度
        grad_g = torch.sum(grad_output * v_normalized, dim=dim, keepdim=True)
        grad_g = torch.clamp(grad_g, min=-1e6, max=1e6)
        
        # 计算v的梯度
        grad_v = grad_output * g
        grad_v = grad_v - v_normalized * torch.sum(grad_v * v_normalized, dim=dim, keepdim=True)
        grad_v = grad_v / (v_norm + 1e-8)
        grad_v = torch.clamp(grad_v, min=-1e6, max=1e6)
        
        return grad_v, grad_g, None

def _weight_norm_interface_forward(v, g, dim):
    return WeightNormFunction.apply(v, g, dim)

# 删除错误的注册代码
# torch.autograd.Function.register_forward_hook(
#     _weight_norm_interface_forward,
#     _weight_norm_interface_backward
# ) 