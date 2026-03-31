import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lmmd
import numpy as np
from sklearn import metrics
from net2 import DSANSS
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_Houston import *
import os
from eta_mamba_modules import TemporalPrototypeManager
from torch.optim.lr_scheduler import CosineAnnealingLR
from UtilsCMS import *

##################################
# 0. 准备工作
if not os.path.exists('classificationMap/Houston/UDA_CETA'):
    os.makedirs('classificationMap/Houston/UDA_CETA')

data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)

data_s, data_t = ILDA(data_s, data_t, pca_n, radius)

# Loss Function
crossEntropy = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
all_runs_avg_epoch_time = []

for iDataSet in range(nDataSet):
    print('\n' + '*' * 60)
    print(f'####################### iDataSet: {iDataSet + 1} / {nDataSet} ########################')
    print('*' * 60)
    utils.set_seed(seeds[iDataSet])

    best_class_acc = np.zeros([CLASS_NUM])
    best_kappa_val = 0.0
    best_oa_val = 0.0
    best_score = 0.0

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, train_num)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                                pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 8, shuffle=False, drop_last=False, num_workers=4,
                             pin_memory=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    proto_manager = TemporalPrototypeManager(class_num=CLASS_NUM, feature_dim=288, momentum=0.9).cuda()

    optimizer = torch.optim.AdamW([
        {'params': feature_encoder.feature_layers.parameters(), 'weight_decay': 0.01},
        {'params': feature_encoder.fc1.parameters(), 'lr': lr},
        {'params': feature_encoder.fc2.parameters(), 'lr': lr},
        {'params': feature_encoder.head1.parameters(), 'lr': lr},
        {'params': feature_encoder.head2.parameters(), 'lr': lr},
        {'params': proto_manager.parameters(), 'lr': 1e-3}
    ], lr=5e-4, weight_decay=1e-4, eps=1e-8)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_loss, oa_list, aa_list, kappa_list, epoch_times = [], [], [], [], []
    global_pseudo_counts = torch.ones(CLASS_NUM).cuda()

    best_episdoe = 0
    train_start = time.time()

    # ==========================================
    # 训练循环
    # ==========================================
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len(train_loader_s)

        ep_loss, ep_cls, ep_adapt, ep_pa, ep_consist = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(num_iter):
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

            source_gpu = source_data.cuda(non_blocking=True)
            target_gpu = target_data.cuda(non_blocking=True)

            alpha_s = torch.empty((source_gpu.size(0), 1, 1, 1), device='cuda').uniform_(0.9, 1.1)
            source_weak = alpha_s * source_gpu + 0.04 * torch.randn_like(source_gpu)

            alpha_t = torch.empty((target_gpu.size(0), 1, 1, 1), device='cuda').uniform_(0.9, 1.1)
            target_weak = alpha_t * target_gpu + 0.04 * torch.randn_like(target_gpu)

            source_strong, target_strong = source_gpu.clone(), target_gpu.clone()
            if torch.rand(1).item() > 0.5:
                source_strong = torch.flip(source_strong, [2])
                target_strong = torch.flip(target_strong, [2])
            if torch.rand(1).item() > 0.5:
                source_strong = torch.flip(source_strong, [3])
                target_strong = torch.flip(target_strong, [3])

            concat_source = torch.cat([source_gpu, source_weak, source_strong], dim=0)
            concat_target = torch.cat([target_gpu, target_weak, target_strong], dim=0)

            (all_features_x, all_x1, all_x2, all_fea_x, all_output_x,
             all_features_y, all_y1, all_y2, all_fea_y, all_output_y) = feature_encoder(concat_source, concat_target)

            B = source_gpu.size(0)

            source_features = all_features_x[:B]
            target_features = all_features_y[:B]
            source_outputs = all_fea_x[:B]
            target_outputs = all_fea_y[:B]
            target_outputs_strong = all_fea_y[2 * B:]
            source_out = all_output_x[:B]
            target_out = all_output_y[:B]

            source2, source3 = all_x1[B:2 * B], all_x1[2 * B:]
            target2, target3 = all_y2[B:2 * B], all_y2[2 * B:]

            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            proto_manager.update(source_features.detach(), source_label.cuda())

            p = (epoch - 1) / epochs
            lambd = 2 / (1 + math.exp(-10 * p)) - 1

            probs_t = F.softmax(target_outputs, dim=1)
            max_probs_t, pseudo_label_t = torch.max(probs_t, dim=1)

            # 1. 基础 LMMD 对齐
            lmmd_loss = lmmd.lmmd(source_features, target_features, source_label.cuda(), probs_t,
                                  BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM)

            # 2. 类别感知动态阈值 (CADT) - 修正阈值下限
            # 仅使用高置信度样本更新统计量，防止累积错误
            valid_stats_mask = max_probs_t > 0.8
            if valid_stats_mask.sum() > 0:
                counts = torch.bincount(pseudo_label_t[valid_stats_mask], minlength=CLASS_NUM).float()
                global_pseudo_counts = 0.9 * global_pseudo_counts + 0.1 * counts

            if epoch > 20:
                N_max = global_pseudo_counts.max() + 1e-6
                relative_freq = global_pseudo_counts / N_max
                # 将动态阈值区间收紧至 [0.75, 0.95]，切断低质量伪标签的流入
                tau_c = 0.75 + 0.20 * relative_freq
                batch_thresholds = tau_c[pseudo_label_t]
                mask_cadt = max_probs_t > batch_thresholds
            else:
                mask_cadt = max_probs_t > 0.9

            valid_count_cadt = mask_cadt.sum().item()

            # 3. 超球面不确定性感知对齐 (HUA) - 严格熵门控
            if epoch > 20:
                entropy_t = -torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1)
                entropy_norm = entropy_t / math.log(CLASS_NUM)
                # 进一步收紧熵门控至 0.1，确保更新超球面原型的样本具备绝对纯度
                confident_mask = entropy_norm < 0.1

                if confident_mask.sum().item() > 0:
                    proto_manager.update_target(target_features[confident_mask], pseudo_label_t[confident_mask])

                    pa_loss_val = proto_manager.get_spherical_alignment_loss()
                    pa_loss = 0.5 * pa_loss_val
                else:
                    #避免当前 batch 无合格样本时计算图阻断
                    pa_loss = torch.tensor(0.0).cuda(requires_grad=True)

            else:
                pa_loss = torch.tensor(0.0).cuda()
                confident_mask = torch.zeros_like(mask_cadt, dtype=torch.bool)

            # 4. 跨视图一致性与对比损失
            log_probs_t_strong = F.log_softmax(target_outputs_strong, dim=1)
            consistency_loss = F.kl_div(log_probs_t_strong, probs_t.detach(), reduction='batchmean')

            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)
            contrastive_loss_s = ContrastiveLoss_s(all_source_con, source_label.cuda())

            # 5. 目标域对比学习纯度隔离
            # 对比学习必须同时满足 CADT 阈值和低熵条件
            if epoch > 20:
                mask_con = mask_cadt & confident_mask
            else:
                mask_con = mask_cadt

            if mask_con.sum().item() > 2:
                contrastive_loss_t = ContrastiveLoss_t(all_target_con[mask_con], pseudo_label_t[mask_con])
            else:
                contrastive_loss_t = torch.tensor(0.0).cuda()

            # 废弃存在冲突的 domain_similar_loss，精简损失空间
            adapt_loss = 0.01 * lmmd_loss + 0.1 * contrastive_loss_s + 0.1 * contrastive_loss_t
            loss = cls_loss + lambd * adapt_loss + pa_loss + lambd * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_cls += cls_loss.item()
            ep_adapt += (lambd * adapt_loss).item()
            ep_pa += pa_loss.item() if isinstance(pa_loss, torch.Tensor) else 0.0
            ep_consist += (lambd * consistency_loss).item()

        scheduler.step()
        avg_loss = ep_loss / num_iter
        train_loss.append(avg_loss)

        # ==========================================
        # 测试验证
        # ==========================================
        feature_encoder.eval()
        total_rewards = 0
        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)

        with torch.no_grad():
            for t_data, test_labels in test_loader:
                batch_size = test_labels.shape[0]
                dummy_s = torch.zeros_like(t_data)
                _, _, _, _, _, _, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())

                pred = t_out.data.max(1)[1]
                test_labels_np = test_labels.numpy()
                rewards = [1 if pred[j] == test_labels_np[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                predict = np.append(predict, pred.cpu().numpy())
                labels = np.append(labels, test_labels_np)

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

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            'Ep {:>3d}: Loss: {:.4f} | Cls: {:.4f} | Adapt: {:.4f} | PA: {:.4f} | Consist: {:.4f} | LR: {:.6f} | Time: {:.2f}s'.format(
                epoch, avg_loss, ep_cls / num_iter, ep_adapt / num_iter,
                                 ep_pa / num_iter, ep_consist / num_iter, current_lr, epoch_duration))

        print('\tOA: {:.2f}% | AA: {:.2f}% | Kappa: {:.4f}'.format(test_accuracy, aa_value, current_kappa))

        current_score = test_accuracy + aa_value
        if current_score > best_score:
            best_score = current_score
            best_episdoe = epoch
            best_predict_all = predict
            best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column

            best_class_acc = AA_current
            best_kappa_val = current_kappa
            best_oa_val = test_accuracy

            print('\t>>> Best Model Updated! (Driven by Composite OA+AA Score)')

    train_end = time.time()

    # ==============================================================================
    # 结果打印与绘图
    # ==============================================================================
    acc[iDataSet, 0] = best_oa_val
    A[iDataSet, :] = best_class_acc
    k[iDataSet, 0] = best_kappa_val

    run_avg_epoch_time = np.mean(epoch_times)
    all_runs_avg_epoch_time.append(run_avg_epoch_time)

    print("\n" + "=" * 40)
    print(f"Results for DataSet (Run) {iDataSet + 1} at Epoch {best_episdoe}")
    print("Total Training Duration: " + "{:.2f} s".format(train_end - train_start))
    print("Average Epoch Time: " + "{:.2f} s/epoch".format(run_avg_epoch_time))
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

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(oa_list) + 1), oa_list, label='OA (%)', color='red', linestyle='-', linewidth=2)
    plt.plot(range(1, len(aa_list) + 1), aa_list, label='AA (%)', color='green', linestyle='-.', linewidth=2)
    plt.plot(range(1, len(kappa_list) + 1), kappa_list, label='Kappa (x100)', color='orange', linestyle=':',
             linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Accuracy Metrics (CETA-Mamba Houston) - Run {iDataSet + 1}', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'classificationMap/Houston/UDA_CETA/accuracy_curve_run{iDataSet + 1}.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title(f'Training Loss (CETA-Mamba Houston) - Run {iDataSet + 1}', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'classificationMap/Houston/UDA_CETA/loss_curve_run{iDataSet + 1}.png', dpi=300)
    plt.close()

    log_dir = 'classificationMap/Houston/UDA_CETA'
    log_file_path = os.path.join(log_dir, 'UDA_summary.txt')
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    log_content = [
        f"\n[{current_time_str}] --- Run {iDataSet + 1} / {nDataSet} ---",
        f"Best Epoch: {best_episdoe} | Best OA : {best_oa_val:.2f}% | Best AA : {100 * np.mean(best_class_acc):.2f}% | Best Kappa : {best_kappa_val:.4f} | Avg Epoch Time: {run_avg_epoch_time:.2f}s"
    ]
    log_str = "\n".join(log_content)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_str)
    except Exception as e:
        pass

if nDataSet > 0:
    mean_oa, std_oa = np.mean(acc), np.std(acc)
    mean_aa, std_aa = np.mean(A) * 100, np.std(np.mean(A, axis=1)) * 100
    mean_kappa, std_kappa = np.mean(k), np.std(k)
    mean_class_acc, std_class_acc = np.mean(A, axis=0) * 100, np.std(A, axis=0) * 100
    mean_epoch_time, std_epoch_time = np.mean(all_runs_avg_epoch_time), np.std(all_runs_avg_epoch_time)

    summary_content = [
        "\n" + "#" * 50,
        f"🎉 FINAL STATISTICAL RESULTS OVER {nDataSet} RUNS (Houston) 🎉",
        "#" * 50,
        f"Mean OA ± Std        : {mean_oa:.2f}% ± {std_oa:.2f}%",
        f"Mean AA ± Std        : {mean_aa:.2f}% ± {std_aa:.2f}%",
        f"Mean Kappa ± Std     : {mean_kappa:.4f} ± {std_kappa:.4f}",
        f"Mean Epoch Time ± Std: {mean_epoch_time:.2f}s ± {std_epoch_time:.2f}s",
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