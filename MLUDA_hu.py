# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset

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
best_predict_all = []
best_acc_all = 0.0

# [新增] 用于存储最佳时刻的各类精度，解决“结果不匹配”问题
best_class_acc = np.zeros([CLASS_NUM])

best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    # 修改后 (使用 from_numpy)
    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
    # delta_phys = torch.zeros(288).cuda()
    proto_manager = TemporalPrototypeManager(class_num=CLASS_NUM, feature_dim=288, momentum=0.9)
    active_queried_indices = []
    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []

    # [新增] 记录详细曲线数据
    oa_list = []  # Overall Accuracy
    aa_list = []  # Average Accuracy
    kappa_list = []

    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        # print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters(), },
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader

        for i in range(1, num_iter):
            source_data, source_label = next(iter_source)
            target_data, target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            # 0
            source_data0 = utils.radiation_noise(source_data)
            source_data0 = source_data0.type(torch.FloatTensor)
            # 1
            source_data1 = utils.flip_augmentation(source_data)
            # 2
            target_data0 = utils.radiation_noise(target_data)
            target_data0 = target_data0.type(torch.FloatTensor)
            # 3
            target_data1 = utils.flip_augmentation(target_data)

            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data.cuda(),
                                                                                        target_data.cuda())
            # mu_s = torch.mean(source_features, dim=0)
            # mu_t = torch.mean(target_features, dim=0)
            # pa_loss = torch.norm(mu_t - (mu_s + delta_phys), p=2)

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # 更新源域原型 & 计算带物理约束的对齐 Loss
            proto_manager.update(source_features.detach(), source_label.cuda())
            pa_loss = proto_manager.get_aligned_loss(target_features, pseudo_label_t)

            (_, source2, _, source_outputs2, _,
             _, _, target2, t1, _) = feature_encoder(source_data0.cuda(), target_data0.cuda())
            (_, source3, _, source_outputs3, _,
             _, _, target3, t2, _) = feature_encoder(source_data1.cuda(), target_data1.cuda())

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            # Loss Lmmd
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            # Loss Con_s
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            # Loss Con_t
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)
            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)

            loss = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss + 0.1 * pa_loss

            # Update parameters
            optimizer.zero_grad()
            loss.backward()

            # [新增] 梯度裁剪：防止梯度爆炸导致 Loss 变 nan
            # max_norm 通常设为 1.0 到 5.0 之间
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            optimizer.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

        # 记录本轮 Loss
        train_loss.append(loss.item())

        print('Epoch {:>3d}: Total Loss: {:6.4f}'.format(epoch, loss.item()))

        train_end = time.time()

        # [关键修改] 每一轮(或者每几轮)都测试一次，这样才能画出曲线
        if epoch % 1 == 0:
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    source_features, source1, _, source_outputs, source_out, test_features, _, _, test_outputs, _ = feature_encoder(
                        Variable(source_data).cuda(), Variable(test_datas).cuda())

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

            test_end = time.time()  # [新增] 修正推理耗时计算错误

            if epoch % 20 == 0 and epoch < epochs:
                print(f">>> Active Learning Query at Epoch {epoch}...")
                feature_encoder.eval()
                all_entropies = []

                # 1. 遍历目标域计算熵
                # 注意：为了下标对齐，这里临时用不打乱的 loader
                eval_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                with torch.no_grad():
                    for t_data, _ in eval_loader:
                        # 构造 dummy source 以便通过 forward
                        dummy_s = torch.zeros_like(t_data)
                        _, _, _, _, _, _, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())

                        probs = F.softmax(t_out, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                        all_entropies.append(entropy.cpu())

                all_entropies = torch.cat(all_entropies)

                # 2. 筛选 Top-K 高熵样本 (最具信息量)
                # 排除已查询过的
                candidate_mask = torch.ones_like(all_entropies, dtype=torch.bool)
                if active_queried_indices:
                    candidate_mask[active_queried_indices] = False

                valid_entropies = all_entropies  # 简化逻辑：在所有样本中找
                # 如果想严格排除，应该只在 candidate_mask 中找，这里为了代码简单，假设 topk 可能包含旧的，下面再过滤

                # [修改方案] 限制最大样本数为 100，避免冲击过大
                limit_percent = int(0.01 * len(test_dataset))
                num_query = min(limit_percent, 100)  # 取 1% 和 100 中的较小值

                if num_query > 0:
                    _, topk_indices = torch.topk(all_entropies, num_query)

                    new_queries = []

                new_queries = []
                for idx in topk_indices.tolist():
                    if idx not in active_queried_indices:
                        new_queries.append(idx)
                        active_queried_indices.append(idx)

                # 3. [闭环关键] 将新样本加入训练集
                if new_queries:
                    print(f"    Added {len(new_queries)} samples to training set.")
                    # 模拟打标：从 target_dataset 中获取真实数据和标签
                    # 注意：这里我们假设 train_dataset 和 test_dataset 都是 TensorDataset
                    # 我们可以直接访问 tensors

                    # 获取原始 Tensor
                    current_source_x = train_loader_s.dataset.tensors[0]
                    current_source_y = train_loader_s.dataset.tensors[1]

                    target_x_all = test_loader.dataset.tensors[0]
                    target_y_all = test_loader.dataset.tensors[1]

                    # 提取新样本
                    query_x = target_x_all[new_queries]
                    query_y = target_y_all[new_queries]  # 使用真实标签 (Oracle)

                    # 拼接到源域数据中
                    new_source_x = torch.cat([current_source_x, query_x], dim=0)
                    new_source_y = torch.cat([current_source_y, query_y], dim=0)

                    # 重新构建 DataLoader
                    new_train_dataset = TensorDataset(new_source_x, new_source_y)
                    # 更新全局的 loader 变量
                    train_loader_s = DataLoader(new_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

                    print(f"    Source Dataset updated: {len(current_source_y)} -> {len(new_source_y)}")
            # 计算本轮 OA
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            oa_list.append(test_accuracy)

            # 计算本轮 AA (Average Accuracy)
            C_current = metrics.confusion_matrix(labels, predict)
            AA_current = np.diag(C_current) / np.sum(C_current, 1, dtype=np.float64)
            aa_value = 100. * np.mean(AA_current)  # AA值
            aa_list.append(aa_value)

            #Kappa 计算
            current_kappa = metrics.cohen_kappa_score(labels, predict)
            kappa_list.append(current_kappa * 100)

            acc[iDataSet] = test_accuracy

            # 记录最后一次的 Kappa 等
            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('\tOA: {:.2f}% | AA: {:.2f}% | Kappa: {:.4f}'.format(test_accuracy, aa_value, current_kappa))

            # [关键修改] 如果发现当前是 Best Epoch，记录下所有的详细数据（包括Class Acc）
            if test_accuracy > last_accuracy:
                # print("save networks for epoch:", epoch)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict  # 保存最好的预测结果用于画图
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column

                # [新增] 专门记录Best时刻的各类精度，用于最后打印
                best_class_acc = AA_current

                print('\t>>> Best Result Updated! Epoch:[{}], Best OA={:.2f}%'.format(best_episdoe, last_accuracy))

            # print('***********************************************************************************')

# 打印最终结果（使用 best_class_acc 而不是最后一轮的）
print("\n" + "=" * 40)
print("Train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("Best OA (Overall Accuracy): " + "{:.2f}%".format(last_accuracy))
print("Best AA (Average Accuracy): " + "{:.2f}%".format(100 * np.mean(best_class_acc)))
print("Best Kappa: " + "{:.4f}".format(k[iDataSet].item()))
print("-" * 40)
print("Accuracy for each class (Matched with Visualization): ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * best_class_acc[i]))

print("=" * 40 + "\n")
# 在原有打印结果的代码处增加：
print("ETA-Mamba 性能报告：")
print("推理耗时 (Latency): " + "{:.5f}".format(test_end - train_end))
# 预期：相比原 MLUDA 的 Transformer/Attention 结构，此处的 Latency 应显著下降
################# classification map ################################

for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# ... (保留你原有的颜色映射代码) ...
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

# 如果需要保存分类图，请取消下面注释
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/Houston/best_map.png")

# ==========================================
# [重点新增] 绘制三合一折线图 (OA, AA, Loss)
# ==========================================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置左侧Y轴 (Accuracy)
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)', color=color)
# 绘制 OA 和 AA
line1 = ax1.plot(range(1, len(oa_list) + 1), oa_list, label='OA (Overall Acc)', color='red', linestyle='-')
line2 = ax1.plot(range(1, len(kappa_list) + 1), kappa_list, label='Kappa (x100)', color='orange', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 100])  # 精度固定在0-100之间

# 设置右侧Y轴 (Loss)
ax2 = ax1.twinx()  # 实例化共享x轴的第二个axes
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
# 绘制 Loss
line3 = ax2.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='blue', alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Training Metrics: Loss, OA & AA')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

save_path = 'classificationMap/Houston/combined_metrics_curve.png'
plt.savefig(save_path, dpi=300)
print(f"Combined metrics curve saved to {save_path}")
plt.close()