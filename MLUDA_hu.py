# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset
# Modified: ETA-Mamba Version (Deep Hybrid + Active Learning + Physical Alignment)

import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
from net2 import DSANSS
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_Houston import *
from sklearn import svm
from UtilsCMS import *
import os
from eta_mamba_modules import TemporalPrototypeManager
from torch.optim.lr_scheduler import LambdaLR

##################################
# 0. 准备工作
if not os.path.exists('classificationMap/Houston'):
    os.makedirs('classificationMap/Houston')

data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)

data_s, data_t = ILDA(data_s, data_t, pca_n, radius)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

# [新增] 专门记录最佳时刻的各项指标
best_class_acc = np.zeros([CLASS_NUM])
best_kappa_val = 0.0
best_oa_val = 0.0

best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

for iDataSet in range(nDataSet):
    print('####################### idataset ######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    # 使用 pin_memory 加速数据传输
    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # 未使用，注释掉
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    # ==========================================
    # 1. 初始化优化器 & 调度器 (移出循环，防止重置)
    # ==========================================
    print("Initializing Optimizer & Prototype Manager...")

    # 物理对齐管理器
    proto_manager = TemporalPrototypeManager(class_num=CLASS_NUM, feature_dim=288, momentum=0.9).cuda()

    # AdamW 优化器：Mamba 最佳拍档
    # 物理参数给予较大的独立学习率 (1e-2)
    optimizer = torch.optim.AdamW([
        {'params': feature_encoder.feature_layers.parameters(), 'weight_decay': 0.01},  # Mamba 层
        {'params': feature_encoder.fc1.parameters(), 'lr': lr},
        {'params': feature_encoder.fc2.parameters(), 'lr': lr},
        {'params': feature_encoder.head1.parameters(), 'lr': lr},
        {'params': feature_encoder.head2.parameters(), 'lr': lr},
        {'params': proto_manager.parameters(), 'lr': 1e-2}  # 物理参数
    ], lr=5e-4, weight_decay=1e-4, eps=1e-8)  # 基础 LR 设为 5e-4


    # Mamba 专用调度器：Warmup (10 Epochs) + Cosine Annealing
    def warmup_cosine_schedule(epoch_idx):
        warmup_epochs = 10
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        else:
            T_max = epochs - warmup_epochs
            curr_T = epoch_idx - warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * curr_T / T_max))


    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    print("Start Training...")

    # 记录曲线数据
    train_loss = []
    oa_list = []
    aa_list = []
    kappa_list = []
    active_queried_indices = []

    last_accuracy = 0.0
    best_episdoe = 0

    # 初始化时间变量
    train_end = 0.0
    test_end = 0.0

    train_start = time.time()

    # ==========================================
    # 2. 训练循环
    # ==========================================
    for epoch in range(1, epochs + 1):
        # 获取当前学习率用于打印 (调度器会在 epoch 末尾 step)
        current_lr = optimizer.param_groups[0]['lr']

        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader
        epoch_loss = 0.0

        # 调试用：记录最后一个 batch 的各部分 loss
        debug_loss_components = {}

        for i in range(1, num_iter):
            try:
                source_data, source_label = next(iter_source)
            except StopIteration:
                iter_source = iter(train_loader_s)
                source_data, source_label = next(iter_source)
            try:
                target_data, target_label = next(iter_target)
            except StopIteration:
                iter_target = iter(train_loader_t)
                target_data, target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            # Augmentation
            source_data0 = utils.radiation_noise(source_data).type(torch.FloatTensor)
            source_data1 = utils.flip_augmentation(source_data)
            target_data0 = utils.radiation_noise(target_data).type(torch.FloatTensor)
            target_data1 = utils.flip_augmentation(target_data)

            # Forward
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data.cuda(),
                                                                                        target_data.cuda())

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # Update Prototypes
            proto_manager.update(source_features.detach(), source_label.cuda())

            # Augmented Forward
            (_, source2, _, source_outputs2, _, _, _, target2, t1, _) = feature_encoder(source_data0.cuda(),
                                                                                        target_data0.cuda())
            (_, source3, _, source_outputs3, _, _, _, target3, t2, _) = feature_encoder(source_data1.cuda(),
                                                                                        target_data1.cuda())

            # Loss Calculation
            cls_loss = crossEntropy(source_outputs, source_label.cuda())

            # 动态权重 lambd (Sigmoid 增长)
            p = (epoch - 1) / epochs
            lambd = 2 / (1 + math.exp(-10 * p)) - 1

            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1),
                                 BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM)

            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)
            contrastive_loss_s = ContrastiveLoss_s(all_source_con, source_label)
            contrastive_loss_t = ContrastiveLoss_t(all_target_con, pseudo_label_t)

            domain_similar_loss = DSH_loss(source_out, target_out)

            # PA Loss (物理对齐) - 增强版
            # 20 Epoch 后介入，权重加大至 0.5，放宽 clamp
            if epoch > 20:
                pa_loss_val = proto_manager.get_aligned_loss(target_features, pseudo_label_t)
                pa_loss = 0.5 * torch.clamp(pa_loss_val, max=20.0)
            else:
                pa_loss = torch.tensor(0.0).cuda()

            # 组合 Loss
            adapt_loss = 0.01 * lmmd_loss + 0.1 * contrastive_loss_s + 0.1 * contrastive_loss_t + 0.1 * domain_similar_loss
            loss = cls_loss + lambd * adapt_loss + pa_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

            # 记录最后一个 batch 的组件供打印
            if i == num_iter - 1:
                debug_loss_components = {
                    'Cls': cls_loss.item(),
                    'Adapt': (lambd * adapt_loss).item(),
                    'PA': pa_loss.item()
                }

        # 更新学习率 (Epoch 结束)
        scheduler.step()

        avg_loss = epoch_loss / num_iter
        train_loss.append(avg_loss)

        # 详细打印
        print('Ep {:>3d}: Avg Loss: {:.4f} | Cls: {:.4f} | Adapt: {:.4f} | PA: {:.4f} | LR: {:.6f}'.format(
            epoch, avg_loss,
            debug_loss_components.get('Cls', 0),
            debug_loss_components.get('Adapt', 0),
            debug_loss_components.get('PA', 0),
            current_lr
        ))

        # 记录训练时间点
        train_end = time.time()

        # ==========================================
        # 3. 测试与主动学习
        # ==========================================
        if epoch % 1 == 0:
            feature_encoder.eval()
            total_rewards = 0
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]
                    # 传入 dummy source data 以适配 net2 forward 接口
                    # 假设 train_dataset[0][0] 形状正确，这里也可以简单构造全0
                    _, _, _, _, _, _, _, _, test_outputs, _ = feature_encoder(source_data.cuda(), test_datas.cuda())

                    pred = test_outputs.data.max(1)[1]
                    test_labels_np = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels_np[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels_np)

            # 记录测试时间点
            test_end = time.time()

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            oa_list.append(test_accuracy)

            current_kappa = metrics.cohen_kappa_score(labels, predict)
            kappa_list.append(current_kappa * 100)

            C_current = metrics.confusion_matrix(labels, predict)
            AA_current = np.diag(C_current) / np.sum(C_current, 1, dtype=np.float64)
            aa_value = 100. * np.mean(AA_current)
            aa_list.append(aa_value)

            print('\tOA: {:.2f}% | AA: {:.2f}% | Kappa: {:.4f}'.format(test_accuracy, aa_value, current_kappa))

            # 仅当 OA 创新高时更新 Best
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column

                best_class_acc = AA_current
                best_kappa_val = current_kappa
                best_oa_val = test_accuracy

                print('\t>>> Best Result Updated!')

                # ==========================================
                # 4. 类别均衡的主动学习 (Class-Balanced Active Learning)
                # ==========================================
                if epoch % 20 == 0 and epoch < epochs:
                    print(f">>> Active Learning Query at Epoch {epoch}...")
                    feature_encoder.eval()
                    all_entropies = []
                    all_preds = []  # [新增] 记录预测类别，用于均衡采样

                    eval_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    with torch.no_grad():
                        for t_data, _ in eval_loader:
                            dummy_s = torch.zeros_like(t_data)
                            _, _, _, _, _, _, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())

                            probs = F.softmax(t_out, dim=1)
                            # 计算熵
                            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                            # 记录预测类别
                            preds = torch.argmax(probs, dim=1)

                            all_entropies.append(entropy.cpu())
                            all_preds.append(preds.cpu())

                    all_entropies = torch.cat(all_entropies)
                    all_preds = torch.cat(all_preds)

                    # 排除已选样本
                    candidate_mask = torch.ones_like(all_entropies, dtype=torch.bool)
                    if active_queried_indices:
                        candidate_mask[active_queried_indices] = False

                    valid_entropies = all_entropies.clone()
                    valid_entropies[~candidate_mask] = -1.0  # 已选或不合法的设为负数

                    # 限制总数量，并计算每个类该分到几个名额
                    limit_percent = int(0.01 * len(test_dataset))
                    num_query_total = min(limit_percent, 100)

                    # [核心修复] 计算每个类分配的查询数量
                    query_per_class = max(1, num_query_total // CLASS_NUM)

                    new_queries = []

                    # 按类别分别去寻找最高熵的样本
                    for c in range(CLASS_NUM):
                        # 找出模型当前预测为类 c，且尚未被查询过的样本索引
                        class_c_mask = (all_preds == c) & candidate_mask
                        class_c_indices = torch.nonzero(class_c_mask).squeeze(-1)

                        if len(class_c_indices) > 0:
                            # 提取这些样本的熵
                            class_c_entropies = valid_entropies[class_c_indices]

                            # 决定当前类实际取几个 (防止该类样本太少不够取)
                            k_c = min(query_per_class, len(class_c_indices))

                            if k_c > 0:
                                # 局部 top-k
                                _, topk_local_idx = torch.topk(class_c_entropies, k_c)
                                # 映射回全局索引
                                topk_global_idx = class_c_indices[topk_local_idx]

                                for idx in topk_global_idx.tolist():
                                    if idx not in active_queried_indices:
                                        new_queries.append(idx)
                                        active_queried_indices.append(idx)

                    # 如果依然有空余名额（某些类样本不足），可以从全局再补齐（可选，这里保持简单，拿到多少是多少）

                    if new_queries:
                        print(f"    Added {len(new_queries)} samples to training set (Class-Balanced).")

                        # 获取 Tensor 数据
                        current_source_x = train_loader_s.dataset.tensors[0]
                        current_source_y = train_loader_s.dataset.tensors[1]
                        target_x_all = test_loader.dataset.tensors[0]
                        target_y_all = test_loader.dataset.tensors[1]

                        query_x = target_x_all[new_queries]
                        query_y = target_y_all[new_queries]

                        # 拼接
                        new_source_x = torch.cat([current_source_x, query_x], dim=0)
                        new_source_y = torch.cat([current_source_y, query_y], dim=0)

                        # 重建 DataLoader
                        new_train_dataset = TensorDataset(new_source_x, new_source_y)
                        train_loader_s = DataLoader(new_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                    drop_last=True, num_workers=4, pin_memory=True)
                        print(f"    Dataset updated: {len(current_source_y)} -> {len(new_source_y)}")

# 打印最终结果
print("\n" + "=" * 40)
# 计算总训练时间 (粗略估计：如果是最后一轮结束时间 - 开始时间)
# 如果需要精确的 pure training time，应该累加 epoch time，这里简单用结束时刻
print("Total Training Duration: " + "{:.2f} s".format(train_end - train_start))
print("Best OA (Overall Accuracy): " + "{:.2f}%".format(best_oa_val))
print("Best AA (Average Accuracy): " + "{:.2f}%".format(100 * np.mean(best_class_acc)))
print("Best Kappa: " + "{:.4f}".format(best_kappa_val))
print("-" * 40)
print("Accuracy for each class (Matched with Visualization): ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * best_class_acc[i]))

print("=" * 40 + "\n")
print("ETA-Mamba 性能报告：")
# Latency: 这里用最后一轮测试的耗时作为参考
print("推理耗时 (Inference Latency): " + "{:.5f} s".format(test_end - train_end))

################# classification map ################################

if best_G is not None:
    for i in range(len(best_predict_all)):
        best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1

    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    # 简单的颜色映射，根据实际情况调整
    colors = [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 1],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0],
        [0.5, 0, 0.5], [0, 0.5, 0.5], [0.3, 0.3, 0.3], [0.8, 0.8, 0.8]
    ]

    for i in range(best_G.shape[0]):
        for j in range(best_G.shape[1]):
            idx = int(best_G[i][j])
            if idx < len(colors):
                hsi_pic[i, j, :] = colors[idx]

    # utils.classification_map(hsi_pic, best_G, 24, "classificationMap/Houston/best_map.png")

# 绘图部分
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)', color=color)
line1 = ax1.plot(range(1, len(oa_list) + 1), oa_list, label='OA', color='red', linestyle='-')
line2 = ax1.plot(range(1, len(kappa_list) + 1), kappa_list, label='Kappa (x100)', color='orange', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 100])

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
line3 = ax2.plot(range(1, len(train_loss) + 1), train_loss, label='Loss', color='blue', alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Training Metrics (ETA-Mamba)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('classificationMap/Houston/combined_metrics_curve.png', dpi=300)
print("Metrics curve saved.")
plt.close()

# ... (上面是你原本的绘图代码 plt.close() 等) ...

# ==========================================
# [新增] 自动保存训练日志 (Log)
# ==========================================
# 定义日志保存路径 (和图片保存在一起)
log_dir = 'classificationMap/Houston/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'training_history_log.txt')

# 获取当前时间
current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# 构建日志内容 (格式与控制台打印的一致)
log_content = []
log_content.append("\n" + "=" * 40)
log_content.append(f"Experiment Time: {current_time_str}")
log_content.append(f"Total Training Duration: {train_end - train_start:.2f} s")
log_content.append(f"Inference Latency: {test_end - train_end:.5f} s")
log_content.append("-" * 40)
log_content.append(f"Best OA : {best_oa_val:.2f}%")
log_content.append(f"Best AA : {100 * np.mean(best_class_acc):.2f}%")
log_content.append(f"Best Kappa : {best_kappa_val:.4f}")
log_content.append("-" * 40)
log_content.append("Accuracy for each class:")
for i in range(CLASS_NUM):
    log_content.append(f"Class {i}: {100 * best_class_acc[i]:.2f}")
log_content.append("=" * 40 + "\n")

# 将内容拼接成字符串
log_str = "\n".join(log_content)

# 以追加模式 ('a') 写入文件
try:
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(log_str)
    print(f"✅ Training log has been appended to: {log_file_path}")
except Exception as e:
    print(f"❌ Failed to write log: {e}")