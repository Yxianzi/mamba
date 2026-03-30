# eta_mamba_modules.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class TemporalPrototypeManager(nn.Module):
    def __init__(self, class_num, feature_dim, momentum=0.9):
        super().__init__()
        self.class_num = class_num
        self.momentum = momentum

        # 注册为 Buffer 保证能被 state_dict 追踪，同时避免参与自动求导导致显存泄漏
        self.register_buffer('prototypes', torch.zeros(class_num, feature_dim))

    def update(self, features, labels):
        # 强制上下文 no_grad 并提前分离计算图，确保纯净
        with torch.no_grad():
            features = features.detach()
            for i in range(self.class_num):
                mask = (labels == i)
                if mask.any():
                    feat_mean = features[mask].mean(0)
                    if self.prototypes[i].sum() == 0:
                        self.prototypes[i] = feat_mean
                    else:
                        self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean