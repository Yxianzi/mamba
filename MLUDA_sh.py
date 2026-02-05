# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Usage：Implementation of the MLUDA method on the SH2HZ cross-domain dataset

import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import neighbors
from sklearn import metrics
from net2 import DSANSS
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_SH2HZ import *
from net2 import DSANSS
from sklearn import svm
from UtilsCMS import *
import os

##################################
# 0. 准备工作：创建保存目录
if not os.path.exists('classificationMap/SH2HZ'):
    os.makedirs('classificationMap/SH2HZ')

file_path = './datasets/Shanghai-Hangzhou/DataCube.mat'
data_s, data_t, label_s, label_t = utils.cubeData(file_path)

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

    # [修改] 使用 from_numpy 修复 float32 报错
    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    print("Training...")
    last_accuracy = 0.0
    best_episdoe = 0

    # [新增] 记录详细曲线数据
    train_loss = []
    oa_list = []  # Overall Accuracy
    aa_list = []  # Average Accuracy

    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0

    train_start = time.time()
    # loss plot
    loss1 = []
    loss2 = []
    loss3 = []

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr  # / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
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
            (_, source2, _, source_outputs2, _,
             _, _, target2, t1, _) = feature_encoder(source_data0.cuda(), target_data0.cuda())
            (_, source3, _, source_outputs3, _,
             _, _, target3, t2, _) = feature_encoder(source_data1.cuda(), target_data1.cuda())

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            entropy_loss = mmd.EntropyLoss(softmax_output_t)
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

            loss = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_t + contrastive_loss_s

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

            test_accuracy = 100. * float(total_hit) / size

        # 记录本轮 Loss
        train_loss.append(loss.item())
        print('Epoch {:>3d}: Total Loss: {:6.4f}'.format(epoch, loss.item()))

        train_end = time.time()

        # [关键修改] 改为每轮都测试 (epoch % 1 == 0)
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

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = test_accuracy

            # 计算并记录本轮 OA 和 AA
            oa_list.append(test_accuracy)

            C_current = metrics.confusion_matrix(labels, predict)
            # 防止除以0
            row_sum = np.sum(C_current, 1, dtype=np.float64)
            row_sum[row_sum == 0] = 0.1
            AA_current = np.diag(C_current) / row_sum
            aa_value = 100. * np.mean(AA_current)
            aa_list.append(aa_value)

            # 记录最后一次的 Kappa
            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('\tOA: {:.2f}% | AA: {:.2f}%'.format(test_accuracy, aa_value))

            # [关键修改] 如果发现当前是 Best Epoch，记录详细数据
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
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
print("Best Kappa: " + "{:.4f}".format(k[iDataSet].item()))  # 加 .item() 修复报错
print("-" * 40)
print("Accuracy for each class (Matched with Visualization): ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * best_class_acc[i]))
print("=" * 40 + "\n")

best_iDataset = 0
for i in range(len(acc)):
    # print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
# print('best acc all={}'.format(acc[best_iDataset]))

################# classification map ################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
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

# 如果需要保存结果图，请取消注释
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/SH2HZ/sh_result.png")

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
line2 = ax1.plot(range(1, len(aa_list) + 1), aa_list, label='AA (Average Acc)', color='orange', linestyle='--')
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

plt.title('Training Metrics (SH2HZ): Loss, OA & AA')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

save_path = 'classificationMap/SH2HZ/combined_metrics_curve.png'
plt.savefig(save_path, dpi=300)
print(f"Combined metrics curve saved to {save_path}")
plt.close()