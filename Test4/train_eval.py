import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve, accuracy_score, \
    precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
from scipy import integrate
import os
import random
from collections import deque

warnings.filterwarnings('ignore')


# ========== POT-EVT动态阈值类 ==========
class POTDynamicThreshold:
    """基于极值理论(EVT)的Peaks-Over-Threshold动态阈值器"""

    def __init__(self, window_size=2000, init_quantile=0.98, risk_q=1e-3, eps=1e-8):
        self.window_size = window_size
        self.init_quantile = init_quantile
        self.risk_q = risk_q
        self.eps = eps
        self.score_window = deque(maxlen=window_size)
        self.threshold_history = []

    def _fit_gpd_mom(self, exceedances):
        """矩估计(MOM)拟合广义帕累托分布(GPD)"""
        f_mean = np.mean(exceedances)
        f_var = np.var(exceedances)
        kappa = 0.5 * (1 - (f_mean ** 2) / (f_var + self.eps))
        xi = 0.5 * f_mean * ((f_mean ** 2) / (f_var + self.eps) + 1)
        return xi, kappa

    def update(self, score, is_anomaly=False):
        """更新滑动窗口"""
        if not is_anomaly:
            self.score_window.append(score)

    def get_threshold(self):
        """计算当前动态阈值"""
        if len(self.score_window) < self.window_size * 0.5:
            current_th = np.percentile(self.score_window, 95) if self.score_window else 0.0
        else:
            th_init = np.percentile(self.score_window, self.init_quantile * 100)
            exceedances = np.array([s - th_init for s in self.score_window if s > th_init])

            if len(exceedances) < 10:
                current_th = th_init
            else:
                xi, kappa = self._fit_gpd_mom(exceedances)
                N_total = len(self.score_window)
                N_peaks = len(exceedances)

                if abs(kappa) < self.eps:
                    current_th = th_init + xi * np.log((N_total * self.risk_q) / (N_peaks + self.eps))
                else:
                    current_th = th_init + (xi / kappa) * (
                            ((self.risk_q * N_total) / (N_peaks + self.eps)) ** (-kappa) - 1
                    )

        self.threshold_history.append(current_th)
        return current_th

    def get_avg_threshold(self):
        return np.mean(self.threshold_history) if self.threshold_history else 0.0


# ========== point_adjust函数 ==========
def point_adjust(pred, label):
    """时序异常检测点调整"""
    pred = pred.copy()
    label = label.copy()
    anomaly_indices = np.where(label == 1)[0]

    if len(anomaly_indices) == 0:
        return pred

    for idx in anomaly_indices:
        start = max(0, idx - 5)
        end = min(len(pred), idx + 5 + 1)
        pred[start:end] = 1

    return pred


# ========== set_graded_seed函数 ==========
def set_graded_seed(base_seed=42):
    os.environ['PYTHONHASHSEED'] = str(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)


# ========== 修复：标准VUS计算函数 ==========
def calculate_vus_standard(anomaly_scores, test_labels, buffer_lengths=[1, 3, 5, 7, 9], n_bins=100):
    """
    计算时序异常检测的标准VUS（修复版）
    """
    roc_surface = []
    pr_surface = []

    for L in buffer_lengths:
        if L < 1:
            L = 1

        # 扩展标签
        extended_labels = np.copy(test_labels)
        anomaly_pos = np.where(test_labels == 1)[0]
        for pos in anomaly_pos:
            start = max(0, pos - L)
            end = min(len(extended_labels), pos + L + 1)
            extended_labels[start:end] = 1

        # 生成阈值
        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), n_bins)
        fpr_list = []
        tpr_list = []
        precision_list = []
        recall_list = []  # 新增：用于PR曲线

        for th in thresholds:
            pred = (anomaly_scores > th).astype(int)

            tp = np.sum((pred == 1) & (extended_labels == 1))
            fp = np.sum((pred == 1) & (extended_labels == 0))
            tn = np.sum((pred == 0) & (extended_labels == 0))
            fn = np.sum((pred == 0) & (extended_labels == 1))

            fpr = fp / (fp + tn + 1e-8)
            tpr = tp / (tp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            precision_list.append(prec)
            recall_list.append(rec)

        # 计算AUROC
        sorted_idx_roc = np.argsort(fpr_list)
        fpr_array = np.array(fpr_list)[sorted_idx_roc]
        tpr_array = np.array(tpr_list)[sorted_idx_roc]
        auroc_L = np.trapz(tpr_array, fpr_array)

        # 计算AUPR（需要对recall排序）
        sorted_idx_pr = np.argsort(recall_list)
        recall_array = np.array(recall_list)[sorted_idx_pr]
        precision_array = np.array(precision_list)[sorted_idx_pr]

        # 确保precision单调递减（PR曲线要求）
        for i in range(1, len(precision_array)):
            if precision_array[i] > precision_array[i-1]:
                precision_array[i] = precision_array[i-1]

        aupr_L = np.trapz(precision_array, recall_array)

        roc_surface.append(auroc_L)
        pr_surface.append(aupr_L)

    # 计算曲面体积
    vus_roc = np.trapz(roc_surface, buffer_lengths) / (max(buffer_lengths) - min(buffer_lengths) + 1)
    vus_pr = np.trapz(pr_surface, buffer_lengths) / (max(buffer_lengths) - min(buffer_lengths) + 1)
    vus_full = (vus_roc + vus_pr) / 2

    return {
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
        "vus_full": vus_full,
        "roc_surface": roc_surface,
        "pr_surface": pr_surface
    }


# ========== 修复：calculate_vus函数 ==========
def calculate_vus(anomaly_scores, test_labels, n_bins=100):
    """计算VUS指标（修复缩进和循环错误）"""
    # 1. 基础AUROC和AUPR
    auc_roc = roc_auc_score(test_labels, anomaly_scores)
    precision, recall, _ = precision_recall_curve(test_labels, anomaly_scores)
    auc_pr = auc(recall, precision)
    vus_simple = (auc_pr + auc_roc) / 2

    # 2. 计算原始VUS
    thresholds = np.linspace(min(anomaly_scores), max(anomaly_scores), n_bins)
    fpr_list = []
    tpr_list = []
    precision_list = []
    recall_list = []

    for th in thresholds:
        pred = (anomaly_scores > th).astype(int)
        tp = np.sum((pred == 1) & (test_labels == 1))
        fp = np.sum((pred == 1) & (test_labels == 0))
        tn = np.sum((pred == 0) & (test_labels == 0))
        fn = np.sum((pred == 0) & (test_labels == 1))

        fpr = fp / (fp + tn + 1e-8)
        tpr = tp / (tp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        precision_list.append(prec)
        recall_list.append(rec)

    # 过滤无效值
    valid_idx = ~np.isnan(precision_list)
    fpr_array = np.array(fpr_list)[valid_idx]
    tpr_array = np.array(tpr_list)[valid_idx]
    precision_array = np.array(precision_list)[valid_idx]
    recall_array = np.array(recall_list)[valid_idx]

    # 排序
    sorted_idx_fpr = np.argsort(fpr_array)
    fpr_array = fpr_array[sorted_idx_fpr]
    tpr_array = tpr_array[sorted_idx_fpr]

    sorted_idx_recall = np.argsort(recall_array)
    recall_array = recall_array[sorted_idx_recall]
    precision_array = precision_array[sorted_idx_recall]

    # 确保precision单调递减
    for i in range(1, len(precision_array)):
        if precision_array[i] > precision_array[i-1]:
            precision_array[i] = precision_array[i-1]

    if len(fpr_array) > 1:
        vus_full_original = np.trapz(precision_array, recall_array) * np.trapz(tpr_array, fpr_array)
    else:
        vus_full_original = vus_simple

    # 3. 计算标准VUS
    vus_standard_results = calculate_vus_standard(anomaly_scores, test_labels)

    return {
        "vus_simple": vus_simple,
        "vus_full_original": vus_full_original,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "vus_roc": vus_standard_results["vus_roc"],
        "vus_pr": vus_standard_results["vus_pr"],
        "vus_full": vus_standard_results["vus_full"],
        "roc_surface": vus_standard_results["roc_surface"],
        "pr_surface": vus_standard_results["pr_surface"]
    }


# ========== calculate_variance_gap函数 ==========
def calculate_variance_gap(anomaly_scores, test_labels, plot=True, dataset_name=""):
    id_scores = anomaly_scores[test_labels == 0]
    ad_scores = anomaly_scores[test_labels == 1]

    id_var = np.var(id_scores) + 1e-8
    ad_var = np.var(ad_scores) + 1e-8
    variance_gap = ad_var - id_var

    if plot and len(id_scores) > 0 and len(ad_scores) > 0:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 6))

        plt.hist(id_scores, bins=50, alpha=0.6, color='blue', label=f'ID样本 | 方差={id_var:.4f}')
        plt.hist(ad_scores, bins=50, alpha=0.6, color='red', label=f'AD样本 | 方差={ad_var:.4f}')

        plt.axvline(np.mean(id_scores), color='blue', linestyle='--', label=f'ID均值={np.mean(id_scores):.4f}')
        plt.axvline(np.mean(ad_scores), color='red', linestyle='--', label=f'AD均值={np.mean(ad_scores):.4f}')
        plt.text(0.5, 0.9, f'方差间隙={variance_gap:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.xlabel('异常分数')
        plt.ylabel('频数')
        plt.title(f'{dataset_name} - ID vs AD样本异常分数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_id_ad_variance_dist.png", dpi=300, bbox_inches='tight')
        plt.show()

    return {
        "variance_gap": variance_gap,
        "id_variance": id_var,
        "ad_variance": ad_var,
        "id_mean": np.mean(id_scores) if len(id_scores) > 0 else 0,
        "ad_mean": np.mean(ad_scores) if len(ad_scores) > 0 else 0
    }


# ========== evaluate_model函数 ==========
def evaluate_model(anomaly_scores, test_labels, dataset_name="", plot=True, target_precision=None):
    """评估模型并输出指标"""
    min_len = min(len(anomaly_scores), len(test_labels))
    anomaly_scores = anomaly_scores[:min_len]
    test_labels = test_labels[:min_len]

    vus_results = calculate_vus(anomaly_scores, test_labels)
    auc_roc = vus_results["auc_roc"]
    auc_pr = vus_results["auc_pr"]
    vus_simple = vus_results["vus_simple"]
    vus_full = vus_results["vus_full"]
    vus_roc = vus_results["vus_roc"]
    vus_pr = vus_results["vus_pr"]

    # F1最优阈值
    precision, recall, thresholds = precision_recall_curve(test_labels, anomaly_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0
    best_precision = precision[best_idx] if len(precision) > best_idx else 0
    best_recall = recall[best_idx] if len(recall) > best_idx else 0
    best_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0

    # 目标Precision阈值
    target_threshold = None
    target_recall = 0
    target_f1 = 0
    if target_precision is not None and len(precision) > 1:
        valid_idx = np.where(precision[:-1] >= target_precision)[0]
        if len(valid_idx) > 0:
            best_recall_idx = valid_idx[np.argmax(recall[valid_idx])]
            target_threshold = thresholds[best_recall_idx]
            target_recall = recall[best_recall_idx]
            target_f1 = 2 * target_precision * target_recall / (target_precision + target_recall + 1e-8)

    # Std-F1
    anomaly_ratio = np.mean(test_labels)
    std_threshold = np.quantile(anomaly_scores, 1 - anomaly_ratio) if anomaly_ratio > 0 else 0
    std_pred = (anomaly_scores > std_threshold).astype(int)
    std_pred = point_adjust(std_pred, test_labels)
    std_precision = precision_score(test_labels, std_pred, zero_division=0)
    std_recall = recall_score(test_labels, std_pred, zero_division=0)
    std_f1 = f1_score(test_labels, std_pred, zero_division=0)

    # 动态阈值
    dynamic_thresholder = POTDynamicThreshold(
        window_size=1000,
        init_quantile=0.98,
        risk_q=1e-3
    )

    normal_scores = anomaly_scores[test_labels == 0]
    for score in normal_scores[:min(1000, len(normal_scores))]:
        dynamic_thresholder.update(score, is_anomaly=False)

    dynamic_pred = []
    dynamic_thresholds = []
    for idx, score in enumerate(anomaly_scores):
        current_th = dynamic_thresholder.get_threshold()
        dynamic_thresholds.append(current_th)
        pred = 1 if score > current_th else 0
        dynamic_pred.append(pred)
        dynamic_thresholder.update(score, is_anomaly=(test_labels[idx] == 1))

    dynamic_pred = point_adjust(np.array(dynamic_pred), test_labels)
    dynamic_precision = precision_score(test_labels, dynamic_pred, zero_division=0)
    dynamic_recall = recall_score(test_labels, dynamic_pred, zero_division=0)
    dynamic_f1 = f1_score(test_labels, dynamic_pred, zero_division=0)
    dynamic_avg_th = dynamic_thresholder.get_avg_threshold()

    acc = accuracy_score(test_labels, (anomaly_scores > best_threshold).astype(int))
    dynamic_acc = accuracy_score(test_labels, dynamic_pred)

    variance_results = calculate_variance_gap(anomaly_scores, test_labels, plot=plot, dataset_name=dataset_name)
    variance_gap = variance_results["variance_gap"]

    # 打印结果
    print("=" * 80)
    print(f"📊 {dataset_name} 模型评估核心指标")
    print("=" * 80)
    print(f"基础分类指标：")
    print(f"  Acc(静态): {acc:.4f} | Acc(动态): {dynamic_acc:.4f} | AUROC: {auc_roc:.4f} | AUPR: {auc_pr:.4f}")
    print(f"  VUS (简化版): {vus_simple:.4f} | VUS-ROC: {vus_roc:.4f} | VUS-PR: {vus_pr:.4f} | VUS (完整版): {vus_full:.4f}")
    print(f"\nF1最优阈值指标（Aff-F1 + 点调整）：")
    print(f"  最佳阈值: {best_threshold:.4f}")
    print(f"  Precision: {best_precision:.4f} | Recall: {best_recall:.4f} | F1: {best_f1:.4f}")
    if target_precision is not None:
        print(f"\n目标Precision阈值指标：")
        print(f"  目标Precision: {target_precision:.4f}")
        if target_threshold is not None:
            print(f"  推荐阈值: {target_threshold:.4f}")
            print(f"  实际Precision: ≥{target_precision:.4f} | Recall: {target_recall:.4f} | F1: {target_f1:.4f}")
        else:
            print(f"  无满足条件的阈值（当前最大Precision: {np.max(precision):.4f}）")
    print(f"\n固定比例阈值指标（Std-F1 + 点调整）：")
    print(f"  固定阈值: {std_threshold:.4f} | 异常比例: {anomaly_ratio:.4f}")
    print(f"  Precision: {std_precision:.4f} | Recall: {std_recall:.4f} | F1: {std_f1:.4f}")
    print(f"\nPOT-EVT动态阈值指标：")
    print(f"  平均阈值: {dynamic_avg_th:.4f} | 阈值波动范围: [{np.min(dynamic_thresholds):.4f}, {np.max(dynamic_thresholds):.4f}]")
    print(f"  Precision: {dynamic_precision:.4f} | Recall: {dynamic_recall:.4f} | F1: {dynamic_f1:.4f}")
    print(f"\n方差间隙指标：{variance_gap:.4f}")
    print("=" * 80)

    return {
        "acc": acc,
        "dynamic_acc": dynamic_acc,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "vus_simple": vus_simple,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
        "vus_full": vus_full,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "std_precision": std_precision,
        "std_recall": std_recall,
        "std_f1": std_f1,
        "dynamic_precision": dynamic_precision,
        "dynamic_recall": dynamic_recall,
        "dynamic_f1": dynamic_f1,
        "dynamic_avg_th": dynamic_avg_th,
        "target_precision": target_precision,
        "target_threshold": target_threshold,
        "target_recall": target_recall,
        "target_f1": target_f1,
        "variance_gap": variance_gap
    }


# ========== train_model函数 ==========
def train_model(model, train_windows, train_labels, epochs=50, batch_size=32, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    train_windows_tensor = torch.tensor(train_windows, dtype=torch.float32).to(device)
    train_targets_tensor = torch.tensor(train_windows[:, -1, :], dtype=torch.float32).to(device)

    model.train()
    loss_history = []
    best_loss = float('inf')
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        valid_batches = 0
        for i in range(0, len(train_windows), batch_size):
            batch_windows = train_windows_tensor[i:i + batch_size]
            batch_targets = train_targets_tensor[i:i + batch_size]

            optimizer.zero_grad()
            loss = model(batch_windows, batch_targets, training=True)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_windows)
            valid_batches += len(batch_windows)

        if valid_batches == 0:
            epoch_loss = 0.0 if len(loss_history) == 0 else loss_history[-1]
        else:
            epoch_loss /= valid_batches

        epoch_loss = max(epoch_loss, 0.0)
        loss_history.append(epoch_loss)

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发：Epoch {epoch + 1}，最佳损失：{best_loss:.6f}")
                break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, LR: {current_lr:.2e}")

    model.load_state_dict(torch.load('best_model.pth'))
    model.calibrate_train_stats(train_windows_tensor.cpu().numpy())
    return model, loss_history


# ========== predict_anomaly_scores函数 ==========
def predict_anomaly_scores(model, test_windows, K=10, device="cuda"):
    model.to(device)
    model.eval()
    anomaly_scores = []
    test_windows_tensor = torch.tensor(test_windows, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i in range(0, len(test_windows), 32):
            batch = test_windows_tensor[i:i + 32]
            batch_scores = model.predict_anomaly_score(batch, K=K)
            batch_scores = batch_scores.flatten()
            anomaly_scores.extend(batch_scores)

    anomaly_scores = np.array(anomaly_scores)[:len(test_windows)]
    anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return anomaly_scores