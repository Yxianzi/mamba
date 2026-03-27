# eta_mamba_modules.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        out_f = self.forward_mamba(x)
        out_b = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        out = torch.cat([out_f, out_b], dim=-1)
        return self.proj(out)


class TemporalPrototypeManager(nn.Module):
    def __init__(self, class_num, feature_dim, momentum=0.9):
        super().__init__()
        self.class_num = class_num
        self.momentum = momentum

        self.register_buffer('prototypes', torch.zeros(class_num, feature_dim))
        # 【修复】：新增全局目标域原型，消除 Batch Size 过小带来的方差震荡
        self.register_buffer('target_prototypes', torch.zeros(class_num, feature_dim))

        self.delta_phi = nn.Parameter(torch.randn(class_num, feature_dim) * 0.01)

    def update(self, features, labels):
        with torch.no_grad():
            for i in range(self.class_num):
                mask = (labels == i)
                if mask.sum() > 0:
                    feat_mean = features[mask].mean(0)
                    if self.prototypes[i].sum() == 0:
                        self.prototypes[i] = feat_mean
                    else:
                        self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean

    # 【新增】：更新高置信度的目标域原型
    def update_target(self, t_features, t_pseudo_labels):
        with torch.no_grad():
            for i in range(self.class_num):
                mask = (t_pseudo_labels == i)
                if mask.sum() > 0:
                    feat_mean = t_features[mask].mean(0)
                    if self.target_prototypes[i].sum() == 0:
                        self.target_prototypes[i] = feat_mean
                    else:
                        self.target_prototypes[i] = self.momentum * self.target_prototypes[i] + (
                                1 - self.momentum) * feat_mean

    # 【修复】：引入物理正则化，不再依赖不稳定且有噪声的 Batch Feature
    def get_aligned_loss(self):
        loss = 0
        count = 0
        for i in range(self.class_num):
            # 只有当源域和目标域该类别的原型都存在时才进行对齐
            if self.target_prototypes[i].sum() != 0 and self.prototypes[i].sum() != 0:
                target_aligned = self.prototypes[i] + self.delta_phi[i]

                # 1. 宏观对齐误差
                align_error = torch.norm(self.target_prototypes[i] - target_aligned, p=2)

                # 2. 【核心修复】：物理偏移正则化 (L2 Penalty)，强制 delta_phi 保持微小，防止吸收所有域差异
                reg_penalty = 0.5 * torch.norm(self.delta_phi[i], p=2)

                loss += (align_error + reg_penalty)
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0).cuda()

        # 【新增创新点】：物理先验初始化

    def init_physical_shift_prior(self, source_global_mean, target_global_mean):
        """
        在训练初期（如第1个Epoch结束时），利用源域和目标域的全局特征统计差异
        作为物理偏移的初始物理先验。
        """
        with torch.no_grad():
            global_shift = target_global_mean - source_global_mean
            # 将全局物理偏移作为所有类别的初始漂移基准
            for i in range(self.class_num):
                self.delta_phi[i].data.copy_(global_shift * 0.1)  # 0.1为缩放因子，防止初始步子太大
