import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ----------------- 1. 基础模块定义 -----------------

class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out_f = self.forward_mamba(x)
        out_b = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        out = self.combine(torch.cat([out_f, out_b], dim=-1))
        return out + residual


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ----------------- 2. 主模型定义 -----------------

class DSANSS(nn.Module):
    def __init__(self, n_band=198, patch_size=3, num_class=3):
        super(DSANSS, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_Mamba(n_band, patch_size, num_class)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)
        self.head1 = nn.Sequential(nn.Linear(288, 128))
        self.head2 = nn.Sequential(nn.Linear(288, 128))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features_x, features_y = self.feature_layers(x, y)

        x1 = F.normalize(self.head1(features_x), dim=1)
        x2 = F.normalize(self.head2(features_x), dim=1)
        fea_x = self.fc1(features_x)
        output_x = self.fc2(features_x)
        output_x = self.sigmoid(output_x)

        features_x = self.dropout(features_x)
        features_y = self.dropout(features_y)

        y1 = F.normalize(self.head1(features_y), dim=1)
        y2 = F.normalize(self.head2(features_y), dim=1)
        fea_y = self.fc1(features_y)
        output_y = self.fc2(features_y)
        output_y = self.sigmoid(output_y)

        return features_x, x1, x2, fea_x, output_x, features_y, y1, y2, fea_y, output_y


class DCRN_Mamba(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_Mamba, self).__init__()
        self.feature_dim = input_channels
        self.sz = patch_size

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

        self.inter_size = 192 + 96

        self.mamba_block = BiSSMBlock(d_model=self.inter_size)

        self.spec_proj_in = nn.Linear(49, 64)
        self.spectral_mamba = BiSSMBlock(d_model=64)
        self.spec_proj_out = nn.Linear(64, 49)

        # 零初始化
        nn.init.zeros_(self.spec_proj_out.weight)
        nn.init.zeros_(self.spec_proj_out.bias)

        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

    def forward_features_cnn(self, x):
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

        out = torch.cat((x1, x2), 1)
        return out

    def forward(self, x, y):
        x_feat = self.forward_features_cnn(x)
        y_feat = self.forward_features_cnn(y)

        B, C, H, W = x_feat.shape
        L = H * W

        # (a) 空间建模阶段
        x_spa = x_feat.view(B, C, L).permute(0, 2, 1)
        y_spa = y_feat.view(B, C, L).permute(0, 2, 1)

        x_spa = self.mamba_block(x_spa)
        y_spa = self.mamba_block(y_spa)

        # (b) 光谱建模阶段
        x_spe = x_spa.permute(0, 2, 1)
        y_spe = y_spa.permute(0, 2, 1)

        x_spe_mid = self.spec_proj_in(x_spe)
        y_spe_mid = self.spec_proj_in(y_spe)

        x_spe_mid = self.spectral_mamba(x_spe_mid)
        y_spe_mid = self.spectral_mamba(y_spe_mid)

        x_spe_out = self.spec_proj_out(x_spe_mid)
        y_spe_out = self.spec_proj_out(y_spe_mid)

        # (c) 旁路残差融合
        x_seq = x_spa + x_spe_out.permute(0, 2, 1)
        y_seq = y_spa + y_spe_out.permute(0, 2, 1)

        x_feat = x_seq.permute(0, 2, 1).view(B, C, H, W)
        y_feat = y_seq.permute(0, 2, 1).view(B, C, H, W)

        # (d) 注意力机制
        x_feat = x_feat.unsqueeze(2)
        y_feat = y_feat.unsqueeze(2)

        x_feat = self.ca(x_feat) * self.sa(x_feat) * x_feat
        y_feat = self.ca(y_feat) * self.sa(y_feat) * y_feat

        x_feat = x_feat.squeeze(2)
        y_feat = y_feat.squeeze(2)

        # (e) 聚合输出
        feat_x = x_feat.mean(dim=(2, 3))
        feat_y = y_feat.mean(dim=(2, 3))

        return feat_x, feat_y