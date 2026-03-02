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
class TemporalPrototypeManager(nn.Module):  # 继承 nn.Module 以便管理参数
    def __init__(self, class_num, feature_dim, momentum=0.9):
        super().__init__()
        self.class_num = class_num
        self.momentum = momentum
        # 源域原型 (不参与梯度更新，由动量更新)
        self.register_buffer('prototypes', torch.zeros(class_num, feature_dim))

        # [关键修改] 季节性演进向量改为“可学习参数”
        # 初始化为微小的随机值，让网络自己去学偏移量
        self.delta_phi = nn.Parameter(torch.randn(class_num, feature_dim) * 0.01)

    def update(self, features, labels):
        # 保持动量更新逻辑不变
        with torch.no_grad():
            for i in range(self.class_num):
                mask = (labels == i)
                if mask.sum() > 0:
                    feat_mean = features[mask].mean(0)
                    # 第一次更新时直接赋值
                    if self.prototypes[i].sum() == 0:
                        self.prototypes[i] = feat_mean
                    else:
                        self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean

    def get_aligned_loss(self, t_features, t_pseudo_labels):
        loss = 0
        count = 0
        # 确保 delta_phi 参与计算图
        for i in range(self.class_num):
            mask = (t_pseudo_labels == i)
            if mask.sum() > 0:
                t_mean = t_features[mask].mean(0)
                # 公式：Target中心 应该接近 (Source中心 + 学习到的物理偏移)
                # 这里 delta_phi 会根据 loss 自动优化
                target_aligned = self.prototypes[i] + self.delta_phi[i]
                loss += torch.norm(t_mean - target_aligned, p=2)
                count += 1
        return loss / count if count > 0 else torch.tensor(0.0).cuda()