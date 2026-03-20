# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Usage：Implementation of the MLUDA method on the SH2HZ cross-domain dataset
# Modified: ETA-Mamba Version (Deep Hybrid + Active Learning + Physical Alignment + Multi-run Stats)

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
from config_SH2HZ import *
from sklearn import svm
from UtilsCMS import *
import os
from eta_mamba_modules import TemporalPrototypeManager
from torch.optim.lr_scheduler import LambdaLR

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

# 用于保存每次循环的最终结果
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

for iDataSet in range(nDataSet):
    print('\n' + '*' * 60)
    print(f'####################### iDataSet: {iDataSet + 1} / {nDataSet} ########################')
    print('*' * 60)
    utils.set_seed(seeds[iDataSet])

    best_class_acc = np.zeros([CLASS_NUM])
    best_kappa_val = 0.0
    best_oa_val = 0.0
    best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    # ==========================================
    # 1. 初始化优化器 & 调度器
    # ==========================================
    print("Initializing Optimizer & Prototype Manager...")
    proto_manager = TemporalPrototypeManager(class_num=CLASS_NUM, feature_dim=288, momentum=0.9).cuda()

    optimizer = torch.optim.AdamW([
        {'params': feature_encoder.feature_layers.parameters(), 'weight_decay': 0.01},
        {'params': feature_encoder.fc1.parameters(), 'lr': lr},
        {'params': feature_encoder.fc2.parameters(), 'lr': lr},
        {'params': feature_encoder.head1.parameters(), 'lr': lr},
        {'params': feature_encoder.head2.parameters(), 'lr': lr},
        {'params': proto_manager.parameters(), 'lr': 1e-2}
    ], lr=5e-4, weight_decay=1e-4, eps=1e-8)


    def warmup_cosine_schedule(epoch_idx):
        warmup_epochs = 10
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        else:
            T_max = epochs - warmup_epochs
            curr_T = epoch_idx - warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * curr_T / max(1, T_max)))


    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    print("Start Training...")

    train_loss = []
    oa_list = []
    aa_list = []
    kappa_list = []
    active_queried_indices = []

    last_accuracy = 0.0
    best_episdoe = 0
    train_end, test_end = 0.0, 0.0
    train_start = time.time()

    # ==========================================
    # 2. 训练循环
    # ==========================================
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader
        epoch_loss = 0.0
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

            source_data0 = utils.radiation_noise(source_data).type(torch.FloatTensor)
            source_data1 = utils.flip_augmentation(source_data)
            target_data0 = utils.radiation_noise(target_data).type(torch.FloatTensor)
            target_data1 = utils.flip_augmentation(target_data)

            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data.cuda(),
                                                                                        target_data.cuda())

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            proto_manager.update(source_features.detach(), source_label.cuda())

            (_, source2, _, source_outputs2, _, _, _, target2, t1, _) = feature_encoder(source_data0.cuda(),
                                                                                        target_data0.cuda())
            (_, source3, _, source_outputs3, _, _, _, target3, t2, _) = feature_encoder(source_data1.cuda(),
                                                                                        target_data1.cuda())

            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            p = (epoch - 1) / epochs
            lambd = 2 / (1 + math.exp(-10 * p)) - 1

            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)
            contrastive_loss_s = ContrastiveLoss_s(all_source_con, source_label)
            contrastive_loss_t = ContrastiveLoss_t(all_target_con, pseudo_label_t)

            if epoch > 20:
                pa_loss_val = proto_manager.get_aligned_loss(target_features, pseudo_label_t)
                pa_loss = 0.2 * torch.log(1 + pa_loss_val)
            else:
                pa_loss = torch.tensor(0.0).cuda()

            # 保持 SH2HZ 特有的 Loss 权重组合
            adapt_loss = 0.01 * lmmd_loss + contrastive_loss_t + contrastive_loss_s
            loss = cls_loss + lambd * adapt_loss + pa_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            if i == num_iter - 1:
                debug_loss_components = {'Cls': cls_loss.item(), 'Adapt': (lambd * adapt_loss).item(),
                                         'PA': pa_loss.item()}

        scheduler.step()
        avg_loss = epoch_loss / num_iter
        train_loss.append(avg_loss)

        print('Ep {:>3d}: Avg Loss: {:.4f} | Cls: {:.4f} | Adapt: {:.4f} | PA: {:.4f} | LR: {:.6f}'.format(
            epoch, avg_loss, debug_loss_components.get('Cls', 0), debug_loss_components.get('Adapt', 0),
            debug_loss_components.get('PA', 0), current_lr))

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
                    _, _, _, _, _, _, _, _, test_outputs, _ = feature_encoder(source_data.cuda(), test_datas.cuda())
                    pred = test_outputs.data.max(1)[1]
                    test_labels_np = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels_np[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)
                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels_np)

            test_end = time.time()
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            oa_list.append(test_accuracy)

            current_kappa = metrics.cohen_kappa_score(labels, predict)
            kappa_list.append(current_kappa * 100)

            C_current = metrics.confusion_matrix(labels, predict)
            row_sum = np.sum(C_current, 1, dtype=np.float64)
            row_sum[row_sum == 0] = 0.1
            AA_current = np.diag(C_current) / row_sum
            aa_value = 100. * np.mean(AA_current)
            aa_list.append(aa_value)

            print('\tOA: {:.2f}% | AA: {:.2f}% | Kappa: {:.4f}'.format(test_accuracy, aa_value, current_kappa))

            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                best_class_acc = AA_current
                best_kappa_val = current_kappa
                best_oa_val = test_accuracy
                print('\t>>> Best Result Updated!')

                # Active Learning (分层类别感知熵策略)
                if epoch % 20 == 0 and epoch < epochs:
                    print(f">>> Active Learning Query at Epoch {epoch}...")
                    feature_encoder.eval()
                    all_entropies = []
                    all_preds = []  # [新增] 必须同时记录预测类别用于分层

                    eval_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    with torch.no_grad():
                        for t_data, _ in eval_loader:
                            dummy_s = torch.zeros_like(t_data)
                            _, _, _, _, _, _, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())
                            probs = F.softmax(t_out, dim=1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                            preds = torch.argmax(probs, dim=1)

                            all_entropies.append(entropy.cpu())
                            all_preds.append(preds.cpu())

                    all_entropies = torch.cat(all_entropies)
                    all_preds = torch.cat(all_preds)

                    # 掩码初始化：排除已采样的样本
                    candidate_mask = torch.ones_like(all_entropies, dtype=torch.bool)
                    if active_queried_indices:
                        candidate_mask[active_queried_indices] = False

                    limit_percent = int(0.01 * len(test_dataset))
                    num_query = min(limit_percent, 100)

                    if num_query > 0:
                        new_queries = []
                        # 1. 计算每个类别的基础配额
                        query_num_per_class = num_query // CLASS_NUM
                        remainder = num_query % CLASS_NUM

                        # 2. 按类别进行独立分层采样
                        for c in range(CLASS_NUM):
                            # 提取当前类别且未被采样的样本掩码
                            class_mask = (all_preds == c) & candidate_mask

                            if class_mask.sum() > 0:
                                class_entropies = all_entropies.clone()
                                # 屏蔽非当前类别的样本
                                class_entropies[~class_mask] = -1.0

                                # 将余数配额分配给前几个类，确保总数严格等于 num_query
                                quota = query_num_per_class + (1 if c < remainder else 0)
                                actual_k = min(quota, class_mask.sum().item())

                                if actual_k > 0:
                                    _, topk_idx = torch.topk(class_entropies, actual_k)
                                    new_queries.extend(topk_idx.tolist())

                        # 3. 极端回退机制：如果某些类被预测的次数少于配额，导致整体采样不足
                        # 则用剩余样本中的全局最高熵来补齐差额
                        if len(new_queries) < num_query:
                            shortage = num_query - len(new_queries)
                            remaining_mask = candidate_mask.clone()
                            remaining_mask[new_queries] = False  # 排除刚才分层已选的

                            fallback_entropies = all_entropies.clone()
                            fallback_entropies[~remaining_mask] = -1.0

                            actual_shortage = min(shortage, remaining_mask.sum().item())
                            if actual_shortage > 0:
                                _, fallback_idx = torch.topk(fallback_entropies, actual_shortage)
                                new_queries.extend(fallback_idx.tolist())

                        # 4. 更新数据集
                        if new_queries:
                            for idx in new_queries:
                                active_queried_indices.append(idx)

                            print(f"    Added {len(new_queries)} samples to training set (Class-aware).")

                            current_source_x = train_loader_s.dataset.tensors[0]
                            current_source_y = train_loader_s.dataset.tensors[1]
                            target_x_all = test_loader.dataset.tensors[0]
                            target_y_all = test_loader.dataset.tensors[1]

                            query_x = target_x_all[new_queries]
                            query_y = target_y_all[new_queries]

                            new_source_x = torch.cat([current_source_x, query_x], dim=0)
                            new_source_y = torch.cat([current_source_y, query_y], dim=0)

                            new_train_dataset = TensorDataset(new_source_x, new_source_y)
                            train_loader_s = DataLoader(new_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                        drop_last=True, num_workers=4, pin_memory=True)

                            print(f"    Dataset updated: {len(current_source_y)} -> {len(new_source_y)}")

    # ---------------------------------------------------------
    # 4. 单次循环结束：保存并打印当次结果
    # ---------------------------------------------------------
    acc[iDataSet, 0] = best_oa_val
    A[iDataSet, :] = best_class_acc
    k[iDataSet, 0] = best_kappa_val

    print("\n" + "=" * 40)
    print(f"Results for DataSet (Run) {iDataSet + 1}")
    print("Total Training Duration: " + "{:.2f} s".format(train_end - train_start))
    print("Best OA (Overall Accuracy): " + "{:.2f}%".format(best_oa_val))
    print("Best AA (Average Accuracy): " + "{:.2f}%".format(100 * np.mean(best_class_acc)))
    print("Best Kappa: " + "{:.4f}".format(best_kappa_val))
    print("-" * 40)
    print("Accuracy for each class:")
    for i in range(CLASS_NUM):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * best_class_acc[i]))
    print("=" * 40 + "\n")

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
    plt.title(f'Training Metrics (ETA-Mamba SH2HZ) - Run {iDataSet + 1}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'classificationMap/SH2HZ/metrics_curve_run{iDataSet + 1}.png', dpi=300)
    plt.close()

    log_dir = 'classificationMap/SH2HZ/Active'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'training_history_log.txt')
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_content = [f"\n[{current_time_str}] --- Run {iDataSet + 1} / {nDataSet} ---",
                   f"Best OA : {best_oa_val:.2f}% | Best AA : {100 * np.mean(best_class_acc):.2f}% | Best Kappa : {best_kappa_val:.4f}"]
    log_str = "\n".join(log_content)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_str)
    except Exception as e:
        pass

# ==============================================================================
# 5. 所有循环结束：计算最终平均值 (Mean) 和 标准差 (Std)，并汇总保存
# ==============================================================================
if nDataSet > 0:
    mean_oa, std_oa = np.mean(acc), np.std(acc)
    mean_aa, std_aa = np.mean(A) * 100, np.std(np.mean(A, axis=1)) * 100
    mean_kappa, std_kappa = np.mean(k), np.std(k)
    mean_class_acc = np.mean(A, axis=0) * 100
    std_class_acc = np.std(A, axis=0) * 100

    summary_content = [
        "\n" + "#" * 50,
        f"🎉 FINAL STATISTICAL RESULTS OVER {nDataSet} RUNS (SH2HZ) 🎉",
        "#" * 50,
        f"Mean OA ± Std    : {mean_oa:.2f}% ± {std_oa:.2f}%",
        f"Mean AA ± Std    : {mean_aa:.2f}% ± {std_aa:.2f}%",
        f"Mean Kappa ± Std : {mean_kappa:.4f} ± {std_kappa:.4f}",
        "-" * 40,
        "Mean Accuracy for each class:"
    ]
    for i in range(CLASS_NUM):
        summary_content.append(f"Class {i}: {mean_class_acc[i]:.2f}% ± {std_class_acc[i]:.2f}%")
    summary_content.append("#" * 50 + "\n")

    summary_str = "\n".join(summary_content)
    print(summary_str)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(summary_str)
    except Exception as e:
        pass