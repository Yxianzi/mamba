# -*- coding:utf-8 -*-
# Modified for ETA-Mamba Speedup

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ... (保持 BiSSMBlock 和其他辅助类不变) ...
class BiSSMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 这里的 d_state 和 d_conv 可以根据显存调整
        self.forward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.backward_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.combine = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (B, L, D)
        # 归一化有助于 Mamba 收敛
        x = self.norm(x)
        out_f = self.forward_mamba(x)
        # 后向 Mamba 需要翻转序列
        out_b = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        return self.combine(torch.cat([out_f, out_b], dim=-1))


class DSANSS(nn.Module):
    # ... (DSANSS 类主体保持不变，主要是 DCRN_Mamba 的调用) ...
    def __init__(self, n_band=198, patch_size=3, num_class=3):
        super(DSANSS, self).__init__()
        self.n_outputs = 288
        # 调用修改后的 DCRN_Mamba
        self.feature_layers = DCRN_Mamba(n_band, patch_size, num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)
        self.head1 = nn.Sequential(nn.Linear(288, 128))
        self.head2 = nn.Sequential(nn.Linear(288, 128))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # ... (保持原有的 forward 逻辑) ...
        features_x, features_y = self.feature_layers(x, y)

        x1 = F.normalize(self.head1(features_x), dim=1)
        x2 = F.normalize(self.head2(features_x), dim=1)
        fea_x = self.fc1(features_x)
        output_x = self.fc2(features_x)
        output_x = self.sigmoid(output_x)

        y1 = F.normalize(self.head1(features_y), dim=1)
        y2 = F.normalize(self.head2(features_y), dim=1)
        fea_y = self.fc1(features_y)
        output_y = self.fc2(features_y)
        output_y = self.sigmoid(output_y)

        return features_x, x1, x2, fea_x, output_x, features_y, y1, y2, fea_y, output_y


class DCRN_Mamba(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_Mamba, self).__init__()
        # ... (前部的 Conv3d 定义保持不变，此处省略以节省篇幅，请保留原代码中的 conv1 到 conv10) ...
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size

        # --- 复制你原有的卷积层定义 ---
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)

        self.inter_size = 192 + 96  # 288

        # Attention Modules
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        # [关键修改] Mamba Backbone
        # 输入维度 288，用于替代池化前的空间特征提取
        self.mamba_backbone = BiSSMBlock(d_model=self.inter_size)

        # 这里的 avgpool 现在仅作为最后的降维手段，或者被 Mamba 的 mean 替代
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

    def forward_features_cnn(self, x):
        # 封装 CNN 部分，避免代码重复
        x = x.unsqueeze(1)
        x1 = self.activation1(self.bn1(self.conv1(x)))
        residual = x1
        x1 = self.activation2(self.bn2(self.conv2(x1)))
        x1 = self.activation3(self.bn3(self.conv3(x1))) + residual

        x1 = self.activation4(self.bn4(self.conv4(x1)))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))

        x2 = self.activation5(self.bn5(self.conv5(x)))
        residual = self.conv8(x2)
        x2 = self.activation6(self.bn6(self.conv6(x2)))
        x2 = self.activation7(self.bn7(self.conv7(x2))) + residual
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))

        out = torch.cat((x1, x2), 1)  # (B, 288, H, W)
        return out

    def forward(self, x, y):
        # 1. 提取基础特征 (3D CNN)
        x_feat = self.forward_features_cnn(x)  # (B, 288, 7, 7)
        y_feat = self.forward_features_cnn(y)  # (B, 288, 7, 7)

        # 2. 空间-通道注意力 (保持轻量级)
        x_feat = self.ca(x_feat) * self.sa(x_feat) * x_feat
        y_feat = self.ca(y_feat) * self.sa(y_feat) * y_feat

        # 3. [核心加速点] Mamba 序列建模
        # 将 (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x_feat.shape
        x_seq = x_feat.view(B, C, -1).permute(0, 2, 1)  # (B, 49, 288)
        y_seq = y_feat.view(B, C, -1).permute(0, 2, 1)

        # Mamba 在序列维度 (49) 上运行，捕捉空间关系
        # 这比先池化成 1x1 再进 Mamba 有效得多
        x_mamba = self.mamba_backbone(x_seq)  # (B, 49, 288)
        y_mamba = self.mamba_backbone(y_seq)

        # 4. 全局平均池化 (聚合空间信息)
        # 替代原本的 self.avgpool
        feat_x = x_mamba.mean(dim=1)  # (B, 288)
        feat_y = y_mamba.mean(dim=1)  # (B, 288)

        return feat_x, feat_y