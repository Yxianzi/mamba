import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math


# ==============================================================================
# 🚀 创新一：全向空间-光谱曼巴 (Omni-Directional Spatial-Spectral Mamba)
# ==============================================================================
class BiSSMBlock(nn.Module):
    """
    重构为 Omni-SSM: 引入 2D 交叉扫描机制 (Cross-Scan)
    注意：为了兼容你原有的 DSANSS 网络，类名保留为 BiSSMBlock
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        # 四个方向的独立 Mamba 扫描器
        self.mamba_h_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)  # 水平正向
        self.mamba_h_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)  # 水平反向
        self.mamba_v_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)  # 垂直正向
        self.mamba_v_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)  # 垂直反向

        # 特征融合投影层
        self.proj = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x shape: (B, L, D)  其中 L = H * W (例如 7x7=49)
        """
        B, L, D = x.shape
        H = int(math.sqrt(L))
        W = H

        # 确保序列长度是一个完美的平方数 (如果是 Patch based)
        assert H * W == L, "OmniSSM requires the sequence length to be a perfect square (H x W)."

        # 1. 恢复到 2D 空间拓扑 (B, H, W, D)
        x_2d = x.view(B, H, W, D)

        # === 扫描 1: 水平正向 (原样展开) ===
        x_h_fwd = x_2d.view(B, L, D)
        out_h_fwd = self.mamba_h_fwd(x_h_fwd)

        # === 扫描 2: 水平反向 (左右翻转) ===
        x_h_bwd = torch.flip(x_2d, dims=[2]).view(B, L, D)
        out_h_bwd = self.mamba_h_bwd(x_h_bwd)
        out_h_bwd = torch.flip(out_h_bwd.view(B, H, W, D), dims=[2]).view(B, L, D)  # 扫完再翻转回来

        # === 扫描 3: 垂直正向 (矩阵转置) ===
        x_v_fwd = x_2d.transpose(1, 2).contiguous().view(B, L, D)
        out_v_fwd = self.mamba_v_fwd(x_v_fwd)
        out_v_fwd = out_v_fwd.view(B, W, H, D).transpose(1, 2).contiguous().view(B, L, D)  # 扫完转置回来

        # === 扫描 4: 垂直反向 (矩阵转置 + 上下翻转) ===
        x_v_bwd = torch.flip(x_2d.transpose(1, 2), dims=[2]).contiguous().view(B, L, D)
        out_v_bwd = self.mamba_v_bwd(x_v_bwd)
        out_v_bwd_2d = torch.flip(out_v_bwd.view(B, W, H, D), dims=[2])
        out_v_bwd = out_v_bwd_2d.transpose(1, 2).contiguous().view(B, L, D)

        # 2. 四向特征全景融合
        out_concat = torch.cat([out_h_fwd, out_h_bwd, out_v_fwd, out_v_bwd], dim=-1)
        out = self.proj(out_concat)

        # 引入残差连接与归一化，稳定梯度
        return self.layer_norm(out + x)


# ==============================================================================
# 🚀 创新二：超球面不确定性感知原型管理器 (Hyperspherical Prototype Alignment)
# ==============================================================================
class TemporalPrototypeManager(nn.Module):
    def __init__(self, class_num, feature_dim, momentum=0.9):
        super().__init__()
        self.class_num = class_num
        self.momentum = momentum

        # 注册中心原型 (无需梯度)
        self.register_buffer('prototypes', torch.zeros(class_num, feature_dim))
        self.register_buffer('target_prototypes', torch.zeros(class_num, feature_dim))

    def update(self, features, labels):
        """源域更新：在超球面上滑动平均"""
        with torch.no_grad():
            # L2 归一化，映射到单位超球面
            features = F.normalize(features.detach(), p=2, dim=1)
            for i in range(self.class_num):
                mask = (labels == i)
                if mask.any():
                    feat_mean = features[mask].mean(0)
                    feat_mean = F.normalize(feat_mean, p=2, dim=0)  # 均值再次归一化
                    if self.prototypes[i].sum() == 0:
                        self.prototypes[i] = feat_mean
                    else:
                        new_proto = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean
                        self.prototypes[i] = F.normalize(new_proto, p=2, dim=0)

    def update_target(self, features, pseudo_labels):
        """目标域更新：只接受高纯度样本"""
        with torch.no_grad():
            features = F.normalize(features.detach(), p=2, dim=1)
            for i in range(self.class_num):
                mask = (pseudo_labels == i)
                if mask.any():
                    feat_mean = features[mask].mean(0)
                    feat_mean = F.normalize(feat_mean, p=2, dim=0)
                    if self.target_prototypes[i].sum() == 0:
                        self.target_prototypes[i] = feat_mean
                    else:
                        new_proto = self.momentum * self.target_prototypes[i] + (1 - self.momentum) * feat_mean
                        self.target_prototypes[i] = F.normalize(new_proto, p=2, dim=0)

    def get_spherical_alignment_loss(self):
        """计算超球面上的余弦对齐误差，天生免疫梯度爆炸"""
        loss = 0.0
        count = 0
        for i in range(self.class_num):
            if self.target_prototypes[i].sum() != 0 and self.prototypes[i].sum() != 0:
                s_proto = self.prototypes[i].detach()
                t_proto = self.target_prototypes[i].detach()

                # 余弦距离: 1 - cos(theta)，范围 [0, 2]
                cos_sim = F.cosine_similarity(s_proto, t_proto, dim=0)
                align_error = 1.0 - cos_sim
                loss = loss + align_error
                count += 1

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=self.prototypes.device, requires_grad=True)