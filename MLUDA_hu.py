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
crossEntropy = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

# 用于保存每次循环(每个 seed)的最终结果
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

for iDataSet in range(nDataSet):
    print('\n' + '*' * 60)
    print(f'####################### iDataSet: {iDataSet + 1} / {nDataSet} ########################')
    print('*' * 60)
    utils.set_seed(seeds[iDataSet])

    # 专门记录当前循环最佳时刻的各项指标
    best_class_acc = np.zeros([CLASS_NUM])
    best_kappa_val = 0.0
    best_oa_val = 0.0
    best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    source_domain_flags = torch.zeros(len(trainY), dtype=torch.int64)
    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long(), source_domain_flags)
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

    train_end = 0.0
    test_end = 0.0

    train_start = time.time()

    # ==========================================
    # 2. 训练循环
    # ==========================================
    for epoch in range(1, epochs + 1):
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
                source_data, source_label, source_domain = next(iter_source)
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
            max_probs_t, pseudo_label_t = torch.max(softmax_output_t, 1)
            conf_mask = max_probs_t > 0.8

            # Update Prototypes
            proto_manager.update(source_features.detach(), source_label.cuda())

            if conf_mask.sum() > 0:
                proto_manager.update_target(target_features.detach()[conf_mask], pseudo_label_t[conf_mask])

            # Augmented Forward
            (_, source2, _, source_outputs2, _, _, _, target2, t1, _) = feature_encoder(source_data0.cuda(),
                                                                                        target_data0.cuda())
            (_, source3, _, source_outputs3, _, _, _, target3, t2, _) = feature_encoder(source_data1.cuda(),
                                                                                        target_data1.cuda())

            # Loss Calculation
            cls_loss = crossEntropy(source_outputs, source_label.cuda())

            p = (epoch - 1) / epochs
            lambd = 2 / (1 + math.exp(-10 * p)) - 1

            pure_source_mask = (source_domain == 0)
            if pure_source_mask.sum() > 2:  # 确保有足够的样本计算分布
                pure_source_features = source_features[pure_source_mask]
                pure_source_labels = source_label[pure_source_mask]

                # 动态计算当前纯源域的 batch size
                pure_batch_size = pure_source_mask.sum().item()

                # 【新增补充修复】：截断 Target 张量，强制对齐 Source 的长度，满足 Weight.py 的严苛要求
                pure_target_features = target_features[:pure_batch_size]
                pure_target_outputs = target_outputs[:pure_batch_size]

                lmmd_loss = mmd.lmmd(pure_source_features, pure_target_features, pure_source_labels.cuda(),
                                     torch.nn.functional.softmax(pure_target_outputs, dim=1),
                                     BATCH_SIZE=pure_batch_size, CLASS_NUM=CLASS_NUM)
            else:
                lmmd_loss = torch.tensor(0.0).cuda()

            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)
            contrastive_loss_s = ContrastiveLoss_s(all_source_con, source_label)
            # 修改对比损失：只对高置信度样本计算
            if conf_mask.sum() > 2:
                contrastive_loss_t = ContrastiveLoss_t(all_target_con[conf_mask], pseudo_label_t[conf_mask])
            else:
                contrastive_loss_t = torch.tensor(0.0).cuda()

            domain_similar_loss = DSH_loss(source_out, target_out)

            # 修改 PA Loss：同样只用高置信度样本对齐
            if epoch > 20:
                pa_loss_val = proto_manager.get_aligned_loss()
                pa_loss = 0.2 * torch.log(1 + pa_loss_val)
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

            if i == num_iter - 1:
                debug_loss_components = {
                    'Cls': cls_loss.item(),
                    'Adapt': (lambd * adapt_loss).item(),
                    'PA': pa_loss.item()
                }

        scheduler.step()

        if epoch == 1:
            # 取当前 epoch 最后一个 batch 的特征中心作为全局分布的近似估计
            source_global = source_features.detach().mean(0)
            target_global = target_features.detach().mean(0)
            proto_manager.init_physical_shift_prior(source_global, target_global)
            print("    >>> [Prior Initialized] Physical Shift Prior has been set based on Epoch 1.")
        # ==========================================================

        avg_loss = epoch_loss / num_iter
        train_loss.append(avg_loss)

        print('Ep {:>3d}: Avg Loss: {:.4f} | Cls: {:.4f} | Adapt: {:.4f} | PA: {:.4f} | LR: {:.6f}'.format(
            epoch, avg_loss,
            debug_loss_components.get('Cls', 0),
            debug_loss_components.get('Adapt', 0),
            debug_loss_components.get('PA', 0),
            current_lr
        ))

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
                for t_data, _ in eval_loader:
                    dummy_s = torch.zeros_like(t_data)
                    # 获取特征和输出
                    _, _, _, _, _, t_feat, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())

                    probs = F.softmax(t_out, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                    preds = torch.argmax(probs, dim=1)

                    # 【新增创新点】：计算目标域样本与源域对应类别原型（Prototypes）的欧式距离
                    # 这代表了该样本的“域漂移程度”
                    shift_distances = torch.zeros(len(preds)).cuda()
                    for c in range(CLASS_NUM):
                        mask = (preds == c)
                        if mask.sum() > 0 and proto_manager.prototypes[c].sum() != 0:
                            # 计算特征到源域原型的距离
                            diff = t_feat[mask] - proto_manager.prototypes[c]
                            shift_distances[mask] = torch.norm(diff, p=2, dim=1)

                    # 将熵和距离归一化后相加，得到最终的 Query Score
                    # 既要模型不确定 (High Entropy)，又要域差异大 (High Domain Shift)
                    norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
                    norm_shift = (shift_distances - shift_distances.min()) / (
                            shift_distances.max() - shift_distances.min() + 1e-8)

                    query_score = norm_entropy + 0.5 * norm_shift  # 0.5 为平衡系数

                    all_scores.append(query_score.cpu())  # 原来的 all_entropies 替换为 all_scores
                    all_preds.append(preds.cpu())

            # 后续的 topk 选取逻辑保持不变，但基于 `all_scores` 而非单纯的熵

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

            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column

                best_class_acc = AA_current
                best_kappa_val = current_kappa
                best_oa_val = test_accuracy

                print('\t>>> Best Result Updated!')

            # Active Learning (领域偏移感知 + 分层类别感知策略)
            if epoch % 20 == 0 and epoch < epochs:
                print(f">>> Active Learning Query at Epoch {epoch}...")
                feature_encoder.eval()

                # 【修复】：这里必须把 all_entropies 改成 all_scores
                all_scores = []
                all_preds = []

                eval_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                with torch.no_grad():
                    for t_data, _ in eval_loader:
                        dummy_s = torch.zeros_like(t_data)
                        _, _, _, _, _, t_feat, _, _, t_out, _ = feature_encoder(dummy_s.cuda(), t_data.cuda())

                        probs = F.softmax(t_out, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                        preds = torch.argmax(probs, dim=1)

                        # 计算目标域样本与源域对应类别原型（Prototypes）的欧式距离 (代表域漂移程度)
                        shift_distances = torch.zeros(len(preds)).cuda()
                        for c in range(CLASS_NUM):
                            mask = (preds == c)
                            if mask.sum() > 0 and proto_manager.prototypes[c].sum() != 0:
                                diff = t_feat[mask] - proto_manager.prototypes[c]
                                shift_distances[mask] = torch.norm(diff, p=2, dim=1)

                        # 归一化并融合分数 (High Entropy + High Domain Shift)
                        norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
                        norm_shift = (shift_distances - shift_distances.min()) / (
                                shift_distances.max() - shift_distances.min() + 1e-8)

                        query_score = norm_entropy + 0.5 * norm_shift  # 综合得分

                        all_scores.append(query_score.cpu())
                        all_preds.append(preds.cpu())

                # 【修复】：拼接列表
                all_scores = torch.cat(all_scores)
                all_preds = torch.cat(all_preds)

                # 掩码初始化
                candidate_mask = torch.ones_like(all_scores, dtype=torch.bool)

                limit_percent = int(0.01 * len(test_dataset))
                num_query = min(limit_percent, 100)

                if num_query > 0:
                    new_queries = []
                    query_num_per_class = num_query // CLASS_NUM
                    remainder = num_query % CLASS_NUM

                    for c in range(CLASS_NUM):
                        class_mask = (all_preds == c) & candidate_mask

                        if class_mask.sum() > 0:
                            # 【修复】：这里原来是 class_entropies = all_entropies.clone()
                            class_scores = all_scores.clone()
                            class_scores[~class_mask] = -1.0  # 屏蔽非当前类别的样本

                            quota = query_num_per_class + (1 if c < remainder else 0)
                            actual_k = min(quota, class_mask.sum().item())

                            if actual_k > 0:
                                _, topk_idx = torch.topk(class_scores, actual_k)
                                new_queries.extend(topk_idx.tolist())

                    # 极端回退机制
                    if len(new_queries) < num_query:
                        shortage = num_query - len(new_queries)
                        remaining_mask = candidate_mask.clone()
                        remaining_mask[new_queries] = False

                        # 【修复】：这里原来是 fallback_entropies
                        fallback_scores = all_scores.clone()
                        fallback_scores[~remaining_mask] = -1.0

                        actual_shortage = min(shortage, remaining_mask.sum().item())
                        if actual_shortage > 0:
                            _, fallback_idx = torch.topk(fallback_scores, actual_shortage)
                            new_queries.extend(fallback_idx.tolist())

                        # 4. 更新数据集 【核心修复：防数据泄露 & 标记目标域样本】
                if new_queries:
                    print(f"    Added {len(new_queries)} samples to training set (Class-aware).")

                    # 获取当前 Loader 中的数据
                    current_source_x = train_loader_s.dataset.tensors[0]
                    current_source_y = train_loader_s.dataset.tensors[1]
                    current_source_d = train_loader_s.dataset.tensors[2]  # 获取现有的域标签

                    target_x_all = test_loader.dataset.tensors[0]
                    target_y_all = test_loader.dataset.tensors[1]

                    # 抽取查询到的数据
                    query_x = target_x_all[new_queries]
                    query_y = target_y_all[new_queries]
                    query_d = torch.ones(len(query_y), dtype=torch.int64)  # 【打上目标域标记 1】

                    # 1. 扩充源域 Loader (带标记)
                    new_source_x = torch.cat([current_source_x, query_x], dim=0)
                    new_source_y = torch.cat([current_source_y, query_y], dim=0)
                    new_source_d = torch.cat([current_source_d, query_d], dim=0)

                    new_train_dataset = TensorDataset(new_source_x, new_source_y, new_source_d)
                    train_loader_s = DataLoader(new_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                drop_last=True, num_workers=4, pin_memory=True)

                    # 2. 从测试集/无监督目标域中【永久剔除】已查询样本，彻底杜绝数据泄露
                    keep_mask = torch.ones(len(target_y_all), dtype=torch.bool)
                    keep_mask[new_queries] = False

                    new_test_x = target_x_all[keep_mask]
                    new_test_y = target_y_all[keep_mask]

                    test_dataset = TensorDataset(new_test_x, new_test_y)

                    # 重新生成测试集和目标域Loader
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             drop_last=True)
                    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                drop_last=True, num_workers=4, pin_memory=True)

                    # 同步更新外部的动态长度
                    len_source_loader = len(train_loader_s)
                    len_target_loader = len(train_loader_t)

                    print(
                        f"    Dataset updated: Source_Loader Size -> {len(new_source_y)} | Test_Loader Size -> {len(new_test_y)}")
    # ---------------------------------------------------------
    # 4. 单次循环结束：保存并打印当次结果
    # ---------------------------------------------------------
    # 【关键修改】：将当次循环的最佳结果存入外部数组，用于后续计算平均值
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

    # 绘图部分 (保存每轮的训练曲线)
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
    plt.title(f'Training Metrics (ETA-Mamba) - Run {iDataSet + 1}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # 为不同组的实验保存不同的图片名，防止被覆盖
    plt.savefig(f'classificationMap/Houston/Active/new1/100epoch Dataset5{iDataSet + 1}.png', dpi=300)
    plt.close()

    # 当次实验的日志记录
    log_dir = 'classificationMap/Houston/Active/new1'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, '100epoch Dataset5.txt')
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    log_content = []
    log_content.append(f"\n[{current_time_str}] --- Run {iDataSet + 1} / {nDataSet} ---")
    log_content.append(
        f"Best OA : {best_oa_val:.2f}% | Best AA : {100 * np.mean(best_class_acc):.2f}% | Best Kappa : {best_kappa_val:.4f}")
    log_str = "\n".join(log_content)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_str)
    except Exception as e:
        pass

# ==============================================================================
# 5. [新增] 所有循环结束：计算最终平均值 (Mean) 和 标准差 (Std)，并汇总保存
# ==============================================================================
if nDataSet > 0:
    # 计算 OA, AA, Kappa 的均值和标准差
    mean_oa = np.mean(acc)
    std_oa = np.std(acc)

    # AA = 每一类准确率的平均值。这里 A 的形状是 (nDataSet, CLASS_NUM)
    mean_aa = np.mean(A) * 100
    # 先算出每次实验的AA，再求这些AA的标准差
    std_aa = np.std(np.mean(A, axis=1)) * 100

    mean_kappa = np.mean(k)
    std_kappa = np.std(k)

    # 计算每一类的均值和标准差
    mean_class_acc = np.mean(A, axis=0) * 100
    std_class_acc = np.std(A, axis=0) * 100

    # 构建最终汇总报告
    summary_content = []
    summary_content.append("\n" + "#" * 50)
    summary_content.append(f"🎉 FINAL STATISTICAL RESULTS OVER {nDataSet} RUNS 🎉")
    summary_content.append("#" * 50)
    summary_content.append(f"Mean OA ± Std    : {mean_oa:.2f}% ± {std_oa:.2f}%")
    summary_content.append(f"Mean AA ± Std    : {mean_aa:.2f}% ± {std_aa:.2f}%")
    summary_content.append(f"Mean Kappa ± Std : {mean_kappa:.4f} ± {std_kappa:.4f}")
    summary_content.append("-" * 40)
    summary_content.append("Mean Accuracy for each class:")
    for i in range(CLASS_NUM):
        summary_content.append(f"Class {i}: {mean_class_acc[i]:.2f}% ± {std_class_acc[i]:.2f}%")
    summary_content.append("#" * 50 + "\n")

    summary_str = "\n".join(summary_content)

    # 1. 打印到控制台
    print(summary_str)

    # 2. 写入到日志文件
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(summary_str)
        print(f"✅ Final summary log has been appended to: {log_file_path}")
    except Exception as e:
        print(f"❌ Failed to write summary log: {e}")
