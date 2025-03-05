from math import pi
import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    """
    对多个张量进行广播拼接（broadcastable concatenation）。

    参数:
        tensors (List[torch.Tensor]): 要拼接的张量列表。
        dim (int): 要拼接的维度，默认为最后一个维度（-1）。

    返回:
        torch.Tensor: 拼接后的张量。

    异常:
        AssertionError: 如果输入的张量维度不一致，或者无法进行广播拼接。
    """
    # 获取张量数量
    num_tensors = len(tensors)
    # 获取每个张量的维度长度，并确保所有张量的维度数量相同
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    # 如果 dim 为负数，则转换为正数索引
    dim = (dim + shape_len) if dim < 0 else dim
    # 获取每个维度上的尺寸，并转置为每个维度的尺寸列表
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    # 找出可以广播的维度（除了拼接维度）
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    # 检查每个可广播维度上的尺寸是否最多有两种不同的值
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    
    # 获取每个可广播维度上的最大尺寸
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    # 为每个张量创建扩展后的尺寸列表
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    # 在拼接维度上插入原始尺寸
    expanded_dims.insert(dim, (dim, dims[dim]))

    # 将尺寸列表转置回每个张量的尺寸
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    # 对每个张量进行扩展，以匹配广播后的形状
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    # 在指定维度上拼接张量
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    """
    对输入张量的最后一个维度进行旋转半周操作。

    参数:
        x (torch.Tensor): 输入张量，最后一个维度的尺寸应为偶数。

    返回:
        torch.Tensor: 旋转后的张量。

    异常:
        AssertionError: 如果最后一个维度的尺寸不是偶数。
    """
    # 重塑张量形状为 (..., d, r)，其中 r=2
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    # 分离出两个部分
    x1, x2 = x.unbind(dim = -1)
    # 旋转半周：交换两个部分，并改变其中一个部分的符号
    x = torch.stack((-x2, x1), dim = -1)
    # 重塑回原始形状
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    """
    视觉旋转嵌入模块，用于在视觉Transformer中引入旋转位置编码。

    参数:
        dim (int): 嵌入的维度。
        pt_seq_len (int): 补丁序列的长度。
        ft_seq_len (int, optional): 要生成的旋转嵌入序列的长度。如果未指定，则默认为 pt_seq_len。
        custom_freqs (torch.Tensor, optional): 自定义频率张量。如果未提供，则根据 freqs_for 参数生成频率。
        freqs_for (str): 频率生成模式，可以是 'lang'（语言）、'pixel'（像素）或 'constant'（常数），默认为 'lang'。
        theta (float): 控制频率分布的参数，默认为10000。
        max_freq (float): 最大频率，默认为10。
        num_freqs (int): 频率的数量，默认为1。
    """
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            # 如果提供了自定义频率，则使用自定义频率
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # 生成语言模式的频率
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            # 生成像素模式的频率
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            # 生成常数模式的频率
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: 
            # 如果未指定 ft_seq_len，则默认为 pt_seq_len
            ft_seq_len = pt_seq_len

        # 生成时间步张量
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # 计算水平方向的频率
        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        # 重复频率以匹配旋转嵌入的维度
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        # 计算垂直方向的频率
        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        # 重复频率以匹配旋转嵌入的维度
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        # 拼接水平和垂直方向的频率
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        # 注册余弦频率缓冲区
        self.register_buffer("freqs_cos", freqs.cos())
        # 注册正弦频率缓冲区
        self.register_buffer("freqs_sin", freqs.sin())

        # 打印频率形状
        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        """
        前向传播函数，应用旋转嵌入。

        参数:
            t (torch.Tensor): 输入张量。
            start_index (int): 开始索引，默认为0。

        返回:
            torch.Tensor: 应用旋转嵌入后的张量。
        """
        # 获取旋转嵌入的维度
        rot_dim = self.freqs_cos.shape[-1]
        # 计算结束索引
        end_index = start_index + rot_dim
        # 检查特征维度是否足够
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        
        # 分离输入张量的不同部分
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        # 应用旋转嵌入
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        # 拼接并返回结果
        return torch.cat((t_left, t, t_right), dim = -1)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    快速视觉旋转嵌入模块，用于在视觉Transformer中高效地引入旋转位置编码。

    参数:
        dim (int): 嵌入的维度。
        pt_seq_len (int, optional): 补丁序列的长度，默认为16。
        clusters (int, optional): 聚类数量，用于分割序列，默认为4。
        custom_freqs (torch.Tensor, optional): 自定义频率张量。如果未提供，则根据 freqs_for 参数生成频率。
        freqs_for (str): 频率生成模式，可以是 'lang'（语言）、'pixel'（像素）或 'constant'（常数），默认为 'lang'。
        theta (float): 控制频率分布的参数，默认为10000。
        max_freq (float): 最大频率，默认为10。
        num_freqs (int): 频率的数量，默认为1。
    """
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        clusters=4,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            # 如果提供了自定义频率，则使用自定义频率
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # 生成语言模式的频率
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            # 生成像素模式的频率
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            # 生成常数模式的频率
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # 设置补丁序列长度
        self.pt_seq_len=pt_seq_len
        # 注册频率缓冲区
        self.register_buffer("freqs", freqs)
        # 设置特征序列长度，假设与补丁序列长度相同
        ft_seq_len = self.pt_seq_len
        # 生成时间步张量
        t = torch.arange(ft_seq_len) / ft_seq_len * self.pt_seq_len

        # 计算频率
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        # 重复频率以匹配旋转嵌入的维度
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        # 拼接水平和垂直方向的频率
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        # 计算余弦频率，并重塑为 (N, C)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        # 计算正弦频率，并重塑为 (N, C)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        # 获取形状
        N, C = freqs_cos.shape
        # 计算高度和宽度，假设 N = H * W
        H = W = int(np.sqrt(N))

        # 重塑余弦频率张量
        freqs_cos=freqs_cos.reshape(2, H//2, 2, W//2,C)
        # 重塑正弦频率张量
        freqs_sin=freqs_sin.reshape(2, H//2, 2, W//2,C)
        # 重排余弦频率张量
        freqs_cos = torch.einsum('hpwqc->hwpqc', freqs_cos).reshape(N, C)
        # 重排正弦频率张量
        freqs_sin = torch.einsum('hpwqc->hwpqc', freqs_sin).reshape(N, C)
        # 注册余弦频率缓冲区
        self.register_buffer('freqs_cos', freqs_cos)
        # 注册正弦频率缓冲区
        self.register_buffer('freqs_sin', freqs_sin)

        # 设置聚类数量
        self.clusters=4
        # 设置序列长度
        self.seq_len=256

    def forward(self, x,scale_index=None): 
        """
        前向传播函数，应用旋转嵌入。

        参数:
            x (torch.Tensor): 输入张量。
            scale_index (int, optional): 缩放索引，用于选择频率范围，默认为None。

        返回:
            torch.Tensor: 应用旋转嵌入后的张量。
        """
        if scale_index is None:
            # 如果没有提供缩放索引，则应用完整的旋转嵌入
            return  x * self.freqs_cos + rotate_half(x) * self.freqs_sin
        
        else:
            # 如果提供了缩放索引，则应用部分旋转嵌入
            return x * self.freqs_cos[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters] + rotate_half(x) * self.freqs_sin[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters]

