#!/usr/bin/env python
# encoding: utf-8

import torch.nn.functional as F
import torch
from Weight import Weight


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    统一在单位超球面上计算高斯核
    """
    # 增加严格的 L2 归一化，统一约束到超球面
    source = F.normalize(source, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


# DSAN - LMMD 核心实现
def lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, CLASS_NUM=7, BATCH_SIZE=32):
    """
    计算局部最大均值差异 (LMMD)
    注意：这里的 Weight.cal_weight 已经过纯张量化优化，直接返回 GPU Tensor
    """
    batch_size = source.size()[0]

    # 纯 GPU 张量计算权重
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, batch_size=BATCH_SIZE, CLASS_NUM=CLASS_NUM)

    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = torch.tensor([0.0], device=source.device)
    if torch.sum(torch.isnan(sum(kernels))):
        return loss

    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    # LMMD 核心公式
    loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)

    return loss