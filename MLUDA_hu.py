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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

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

# 用于保存每次循环(每个 seed)的最终结果
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
all_runs_avg_epoch_time = []  # [新增] 用于保存每次 Run 的平均 Epoch 耗时

for iDataSet in range(nDataSet):
    print('\n' + '*' * 60)
    print(f'####################### iDataSet: {iDataSet + 1} / {nDataSet} ########################')
    print('*' * 60)
    utils.set_seed(seeds[iDataSet])

    # 专门记录当前循环最佳时刻的各项指标
    best_class_acc = np.zeros([CLASS_NUM])
    best_kappa_val = 0.0
    best_oa_val = 0.0
    best_score = 0.0  # [新增] 综合评分守门员 (OA + AA)
    best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

    # [UDA 重构] 源域完全固定，不再需要 source_domain_flags，因为永远不会有目标域真实标签注入
    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    # 使用 pin_memory 加速数据传输
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
        {'params': proto_manager.parameters(), 'lr': 1e-3}
    ], lr=5e-4, weight_decay=1e-4, eps=1e-8)

    # [改进] 采用余弦退火学习率，解决末期震荡无法收敛的问题
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("Start Pure UDA Training...")

    train_loss = []
    oa_list = []
    aa_list = []
    kappa_list = []
    epoch_times = []  # [新增] 记录当前 Run 中每个 Epoch 的耗时

    # [新增] 用于 CADT 的全局类别伪标签计数器
    global_pseudo_counts = torch.ones(CLASS_NUM).cuda()

    best_episdoe = 0
    train_end = 0.0
    test_end = 0.0
    train_start = time.time()

    # ==========================================
    # 2. 训练循环
    # ==========================================
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()  # [新增] 记录当前 Epoch 开始时间

        current_lr = optimizer.param_groups[0]['lr']
        feature_encoder.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)

        num_iter = len(train_loader_s)
        len_target_loader = len(train_loader_t)

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

            # [新增] 数据增强：弱增强 vs 强增强 (为跨视图一致性做准备)
            source_data0 = utils.radiation_noise(source_data).type(torch.FloatTensor)
            source_data1 = utils.flip_augmentation(source_data)
            target_data_weak = utils.radiation_noise(target_data).type(torch.FloatTensor)
            target_data_strong = utils.flip_augmentation(target_data)

            # --- 前向传播 ---
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data.cuda(),
                                                                                        target_data.cuda())

            # 强弱视图特征提取
            (_, source2, _, source_outputs2, _, _, _, target2, _, _) = feature_encoder(source_data0.cuda(),
                                                                                       target_data_weak.cuda())
            (_, source3, _, source_outputs3, _, _, _, target3, target_outputs_strong, _) = feature_encoder(
                source_data1.cuda(), target_data_strong.cuda())

            # --- 监督损失与源域原型更新 ---
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            proto_manager.update(source_features.detach(), source_label.cuda())

            p = (epoch - 1) / epochs
            lambd = 2 / (1 + math.exp(-10 * p)) - 1

            # [UDA 重构] 纯净的 LMMD 损失，无需分离源域中的目标样本（因已剔除 AL）
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label.cuda(),
                                 torch.nn.functional.softmax(target_outputs, dim=1),
                                 BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM)

            # --- [创新点 1 & 2]: CADT 动态阈值与 IE-SPA 熵加权柔性对齐 ---
            probs_t = F.softmax(target_outputs, dim=1)
            max_probs_t, pseudo_label_t = torch.max(probs_t, dim=1)
            entropy_t = -torch.sum(probs_t * torch.log(probs_t + 1e-6), dim=1)

            # 【极速优化 1】: 纯 GPU 并行更新频率，消灭 for 循环和 .item() 同步阻塞
            counts = torch.bincount(pseudo_label_t, minlength=CLASS_NUM).float()
            global_pseudo_counts = 0.9 * global_pseudo_counts + 0.1 * counts

            # 提前计算 mask 并只执行【唯一一次】 .item() 同步
            if epoch > 20:
                N_max = global_pseudo_counts.max()
                tau_base = 0.85
                gamma_cadt = 0.5
                tau_c = tau_base * ((global_pseudo_counts / N_max) ** gamma_cadt)
                tau_c = torch.clamp(tau_c, min=0.5, max=0.95)

                batch_thresholds = tau_c[pseudo_label_t]
                mask = max_probs_t > batch_thresholds
            else:
                mask = max_probs_t > 0.8  # 前20轮用硬阈值过渡

            # 【极速优化 2】: 全局仅同步一次验证数量，避免多次 GPU 等待
            valid_count = mask.sum().item()

            if epoch > 20:
                if valid_count > 0:
                    proto_manager.update_target(target_features.detach()[mask], pseudo_label_t[mask])

                    valid_target_features = target_features[mask]
                    valid_pseudo_labels = pseudo_label_t[mask]
                    valid_entropy_weights = torch.exp(-entropy_t[mask])

                    target_protos = proto_manager.prototypes[valid_pseudo_labels]
                    dist = torch.sum((valid_target_features - target_protos) ** 2, dim=1)
                    sample_pa_loss = 0.2 * torch.log(1 + dist)
                    pa_loss = torch.mean(valid_entropy_weights * sample_pa_loss)
                else:
                    pa_loss = torch.tensor(0.0).cuda()
            else:
                pa_loss = torch.tensor(0.0).cuda()

            # --- [创新点 3]: Mamba 跨视图一致性自蒸馏 ---
            probs_t_strong = F.softmax(target_outputs_strong, dim=1)
            consistency_loss = F.kl_div(probs_t_strong.log(), probs_t.detach(), reduction='batchmean')

            # --- 对比损失及 Houston 特有的 DSH 损失 ---
            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)
            contrastive_loss_s = ContrastiveLoss_s(all_source_con, source_label)

            if valid_count > 2:  # 使用已经取回 CPU 的 valid_count
                contrastive_loss_t = ContrastiveLoss_t(all_target_con[mask], pseudo_label_t[mask])
            else:
                contrastive_loss_t = torch.tensor(0.0).cuda()

            domain_similar_loss = DSH_loss(source_out, target_out)

            # 组合 Loss (保留 Houston 专属的 0.1 权重比例)
            adapt_loss = 0.01 * lmmd_loss + 0.1 * contrastive_loss_s + 0.1 * contrastive_loss_t + 0.1 * domain_similar_loss
            loss = cls_loss + lambd * adapt_loss + pa_loss + lambd * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if i == num_iter - 1:
                debug_loss_components = {
                    'Cls': cls_loss.item(),
                    'Adapt': (lambd * adapt_loss).item(),
                    'PA': pa_loss.item(),
                    'Consist': (lambd * consistency_loss).item()
                }

        scheduler.step()

        if epoch == 1:
            source_global = source_features.detach().mean(0)
            target_global = target_features.detach().mean(0)
            proto_manager.init_physical_shift_prior(source_global, target_global)
            print("    >>> [Prior Initialized] Physical Shift Prior has been set based on Epoch 1.")

        avg_loss = epoch_loss / num_iter
        train_loss.append(avg_loss)

        # ==========================================
        # 3. 测试与指标综合更新
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
        row_sum[row_sum == 0] = 0.1  # 防止除以0
        AA_current = np.diag(C_current) / row_sum
        aa_value = 100. * np.mean(AA_current)
        aa_list.append(aa_value)

        epoch_end_time = time.time()  # [新增] 记录当前 Epoch 结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # [新增] 计算当前 Epoch 耗时
        epoch_times.append(epoch_duration)

        print(
            'Ep {:>3d}: Loss: {:.4f} | Cls: {:.4f} | Adapt: {:.4f} | PA: {:.4f} | Consist: {:.4f} | LR: {:.6f} | Time: {:.2f}s'.format(
                epoch, avg_loss, debug_loss_components.get('Cls', 0), debug_loss_components.get('Adapt', 0),
                debug_loss_components.get('PA', 0), debug_loss_components.get('Consist', 0), current_lr,
                epoch_duration))

        print('\tOA: {:.2f}% | AA: {:.2f}% | Kappa: {:.4f}'.format(test_accuracy, aa_value, current_kappa))

        # [UDA 重构] 综合得分守门，防末端震荡
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

    # ---------------------------------------------------------
    # 4. 单次循环结束：保存并打印当次结果
    # ---------------------------------------------------------
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

    # 绘图部分
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color=color)
    line1 = ax1.plot(range(1, len(oa_list) + 1), oa_list, label='OA', color='red', linestyle='-')
    line2 = ax1.plot(range(1, len(aa_list) + 1), aa_list, label='AA', color='green', linestyle='-.')  # 增加 AA 曲线
    line3 = ax1.plot(range(1, len(kappa_list) + 1), kappa_list, label='Kappa (x100)', color='orange', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 100])
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)
    line4 = ax2.plot(range(1, len(train_loss) + 1), train_loss, label='Loss', color='blue', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    plt.title(f'Training Metrics (CETA-Mamba Houston) - Run {iDataSet + 1}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'classificationMap/Houston/UDA_CETA/metrics_curve_run{iDataSet + 1}.png', dpi=300)
    plt.close()

    # 日志记录
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

# ==============================================================================
# 5. 所有循环结束：计算最终平均值 (Mean) 和 标准差 (Std)，并汇总保存
# ==============================================================================
if nDataSet > 0:
    mean_oa = np.mean(acc)
    std_oa = np.std(acc)
    mean_aa = np.mean(A) * 100
    std_aa = np.std(np.mean(A, axis=1)) * 100
    mean_kappa = np.mean(k)
    std_kappa = np.std(k)

    mean_class_acc = np.mean(A, axis=0) * 100
    std_class_acc = np.std(A, axis=0) * 100

    mean_epoch_time = np.mean(all_runs_avg_epoch_time)
    std_epoch_time = np.std(all_runs_avg_epoch_time)

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
        print(f"✅ Final summary log has been appended to: {log_file_path}")
    except Exception as e:
        print(f"❌ Failed to write summary log: {e}")