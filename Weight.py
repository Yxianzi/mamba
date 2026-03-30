import torch
import torch.nn.functional as F

class Weight:
    @staticmethod
    def cal_weight(s_label, t_label, batch_size, CLASS_NUM):
        """
        纯 GPU 张量化计算 LMMD 权重，消灭 NumPy 和 CPU 阻塞
        s_label: 源域真实标签 [batch_size]
        t_label: 目标域预测概率 [batch_size, CLASS_NUM]
        """
        # 1. 源域 One-hot 编码与归一化
        s_vec_label = F.one_hot(s_label, num_classes=CLASS_NUM).float() # [batch_size, CLASS_NUM]
        s_sum = s_vec_label.sum(dim=0, keepdim=True)
        s_sum[s_sum == 0] = 100.0
        s_vec_label = s_vec_label / s_sum

        # 2. 目标域概率分布归一化
        t_vec_label = t_label.float()
        t_sum = t_vec_label.sum(dim=0, keepdim=True)
        t_sum[t_sum == 0] = 100.0
        t_vec_label = t_vec_label / t_sum

        # 3. 筛选在源域和目标域中同时存在的类别
        t_sca_label = t_label.max(dim=1)[1]
        s_present = torch.bincount(s_label, minlength=CLASS_NUM) > 0
        t_present = torch.bincount(t_sca_label, minlength=CLASS_NUM) > 0
        valid_classes = s_present & t_present  # [CLASS_NUM] 的布尔掩码

        # 4. 利用掩码屏蔽无效类别的特征列
        s_vec_valid = s_vec_label * valid_classes.unsqueeze(0)
        t_vec_valid = t_vec_label * valid_classes.unsqueeze(0)

        # 5. 矩阵乘法直接计算权重矩阵 (替代原来的 for 循环累加)
        weight_ss = s_vec_valid @ s_vec_valid.T
        weight_tt = t_vec_valid @ t_vec_valid.T
        weight_st = s_vec_valid @ t_vec_valid.T

        count = valid_classes.sum().float()

        # 6. 平均操作
        if count > 0:
            weight_ss = weight_ss / count
            weight_tt = weight_tt / count
            weight_st = weight_st / count
        else:
            weight_ss = torch.zeros((batch_size, batch_size), device=s_label.device)
            weight_tt = torch.zeros((batch_size, batch_size), device=s_label.device)
            weight_st = torch.zeros((batch_size, batch_size), device=s_label.device)

        return weight_ss, weight_tt, weight_st