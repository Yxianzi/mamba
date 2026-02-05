# eta_mamba_modules.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba # 需安装 mamba-ssm

# 创新点一：双向空间-光谱 Mamba (Bi-SSM)
class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x shape: (B, L, D) -> L 为波段*空间序列长度
        out_f = self.forward_mamba(x)
        out_b = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        out = torch.cat([out_f, out_b], dim=-1)
        return self.proj(out)

# 创新点二：时序动量原型管理器
class TemporalPrototypeManager:
    def __init__(self, class_num, feature_dim, momentum=0.9):
        self.class_num = class_num
        self.momentum = momentum
        # 初始化源域原型
        self.prototypes = torch.zeros(class_num, feature_dim).cuda()
        # 季节性演进向量 (假设根据先验或均值差初始化)
        self.delta_phi = torch.zeros(class_num, feature_dim).cuda()

    @torch.no_grad()
    def update(self, features, labels):
        for i in range(self.class_num):
            mask = (labels == i)
            if mask.sum() > 0:
                feat_mean = features[mask].mean(0)
                self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean

    def get_aligned_loss(self, t_features, t_pseudo_labels):
        loss = 0
        count = 0
        for i in range(self.class_num):
            mask = (t_pseudo_labels == i)
            if mask.sum() > 0:
                t_mean = t_features[mask].mean(0)
                # 应用公式：Loss = || mu_t - (mu_s + delta_phi) ||^2
                loss += torch.norm(t_mean - (self.prototypes[i] + self.delta_phi[i]), p=2)
                count += 1
        return loss / count if count > 0 else torch.tensor(0.0).cuda()