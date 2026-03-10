import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')


class LightweightUSAD(nn.Module):
    """
    轻量化USAD模型（UnSupervised Anomaly Detection）
    双解码器对抗训练，适配时序窗口异常检测
    """

    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        # 编码器（轻量化全连接结构）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # 解码器1（重构正常数据）
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        # 解码器2（对抗训练，突出异常）
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        """
        前向传播（适配时序窗口数据）
        :param x: 输入张量，shape=(batch_size, window_length, input_dim)
        :return: recon1, recon2: 两个解码器的重构结果
        """
        batch_size, window_len, input_dim = x.shape
        # 展平窗口内的特征：(batch_size*window_len, input_dim)
        x_flat = x.reshape(batch_size * window_len, input_dim)

        # 编码+双解码
        z = self.encoder(x_flat)
        x_recon1 = self.decoder1(z)
        x_recon2 = self.decoder2(z)

        # 恢复时序窗口形状
        x_recon1 = x_recon1.reshape(batch_size, window_len, input_dim)
        x_recon2 = x_recon2.reshape(batch_size, window_len, input_dim)
        return x_recon1, x_recon2


class USADAnomalyDetector:
    """
    USAD异常检测器封装（适配时序窗口数据）
    修复核心问题：对抗训练逻辑、异常分数归一化、训练集过滤
    """

    def __init__(self, input_dim, latent_dim=32, lr=1e-4, device="cpu", random_state=42):
        """
        初始化USAD检测器
        :param input_dim: 单步特征维度（非窗口展平维度）
        :param latent_dim: 隐层维度
        :param lr: 学习率
        :param device: 运行设备（cuda/cpu）
        :param random_state: 随机种子
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.device = torch.device(device)
        self.random_state = random_state

        # 固定随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # 初始化模型
        self.model = LightweightUSAD(input_dim, latent_dim).to(self.device)

        # 分离优化器（对抗训练核心：不同阶段优化不同参数）
        self.optimizer_enc_dec1 = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=lr, weight_decay=1e-6
        )
        self.optimizer_dec2 = optim.Adam(
            list(self.model.decoder2.parameters()),
            lr=lr, weight_decay=1e-6
        )
        self.criterion = nn.MSELoss(reduction='mean')

        # 记录训练损失
        self.loss_history1 = []
        self.loss_history2 = []

        # 存储训练集统计信息（用于分数归一化）
        self.train_mse1_mean = 0.0
        self.train_mse1_std = 1.0
        self.train_mse2_mean = 0.0
        self.train_mse2_std = 1.0

    def _filter_normal_train_data(self, train_windows, train_labels=None):
        """过滤训练集：仅保留正常样本（USAD核心要求）"""
        if train_labels is None:
            return train_windows  # 无标签时默认全部为正常
        normal_mask = train_labels == 0
        # 窗口级标签：只要窗口内有异常，就过滤掉
        if len(train_labels.shape) == 2:  # (n_windows, window_length)
            normal_mask = np.all(normal_mask, axis=1)
        normal_train = train_windows[normal_mask]
        return normal_train

    def train(self, train_windows, train_labels=None, epochs=50, batch_size=16, verbose=True):
        """
        训练USAD模型（修复对抗训练逻辑）
        :param train_windows: 训练时序窗口，shape=(n_windows, window_length, input_dim)
        :param train_labels: 训练标签（用于过滤异常样本）
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param verbose: 是否打印训练日志
        :return: 损失历史
        """
        # 核心修复1：过滤训练集，仅保留正常样本
        train_windows = self._filter_normal_train_data(train_windows, train_labels)
        if len(train_windows) == 0:
            raise ValueError("训练集过滤后无正常样本！")

        # 数据转换
        train_tensor = torch.FloatTensor(train_windows).to(self.device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        self.loss_history1 = []
        self.loss_history2 = []

        if verbose:
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"📚 USAD训练数据加载完成 | 正常样本数：{len(train_windows)} | 可训练参数：{param_count:,}")
            print(f"📈 开始对抗训练 | epochs={epochs}, batch_size={batch_size}, lr={self.lr}")

        for epoch in range(epochs):
            total_loss1 = 0.0
            total_loss2 = 0.0

            for batch in train_loader:
                x = batch[0].to(self.device)

                # 阶段1：训练编码器+解码器1（重构损失最小化）
                self.optimizer_enc_dec1.zero_grad()
                recon1, recon2 = self.model(x)
                loss1 = self.criterion(recon1, x)  # 仅优化重构损失
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 调整梯度裁剪
                self.optimizer_enc_dec1.step()

                # 阶段2：训练解码器2（对抗训练，修复损失逻辑）
                self.optimizer_dec2.zero_grad()
                recon1, recon2 = self.model(x)
                # 核心修复2：USAD原版Loss2（max(0, 重构损失 - 正则项)）
                loss2 = torch.maximum(
                    self.criterion(recon2, x) - 0.1 * self.criterion(recon1, x),
                    torch.tensor(0.0, device=self.device)
                )
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.model.decoder2.parameters(), max_norm=1.0)
                self.optimizer_dec2.step()

                total_loss1 += loss1.item() * x.size(0)
                total_loss2 += loss2.item() * x.size(0)

            # 计算平均损失
            avg_loss1 = total_loss1 / len(train_loader.dataset)
            avg_loss2 = total_loss2 / len(train_loader.dataset)
            self.loss_history1.append(avg_loss1)
            self.loss_history2.append(avg_loss2)

            # 打印训练日志
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss1: {avg_loss1:.6f} | Loss2: {avg_loss2:.6f}")

        # 核心修复3：计算训练集MSE统计（用于测试集分数归一化）
        self._compute_train_mse_stats(train_windows)

        if verbose:
            print(f"✅ USAD训练完成 | 最后一轮Loss1: {self.loss_history1[-1]:.6f} | Loss2: {self.loss_history2[-1]:.6f}")

        return self.loss_history1, self.loss_history2

    def _compute_train_mse_stats(self, train_windows):
        """计算训练集MSE的均值/标准差（用于分数归一化）"""
        self.model.eval()
        train_tensor = torch.FloatTensor(train_windows).to(self.device)
        with torch.no_grad():
            recon1, recon2 = self.model(train_tensor)
            mse1 = torch.mean((recon1 - train_tensor) ** 2, dim=(1, 2))
            mse2 = torch.mean((recon2 - train_tensor) ** 2, dim=(1, 2))

        self.train_mse1_mean = mse1.mean().cpu().numpy()
        self.train_mse1_std = mse1.std().cpu().numpy() + 1e-8  # 避免除0
        self.train_mse2_mean = mse2.mean().cpu().numpy()
        self.train_mse2_std = mse2.std().cpu().numpy() + 1e-8

    def predict(self, test_windows):
        """
        预测异常分数（修复：训练集归一化，避免分数失控）
        :param test_windows: 测试时序窗口，shape=(n_windows, window_length, input_dim)
        :return: anomaly_scores: 归一化后的异常分数数组
        """
        self.model.eval()
        test_tensor = torch.FloatTensor(test_windows).to(self.device)

        with torch.no_grad():
            recon1, recon2 = self.model(test_tensor)
            # 计算窗口级MSE
            mse1 = torch.mean((recon1 - test_tensor) ** 2, dim=(1, 2))
            mse2 = torch.mean((recon2 - test_tensor) ** 2, dim=(1, 2))

            # 核心修复4：用训练集统计归一化MSE（USAD原版逻辑）
            mse1_norm = (mse1 - self.train_mse1_mean) / self.train_mse1_std
            mse2_norm = (mse2 - self.train_mse2_mean) / self.train_mse2_std

            # 异常分数：加权和（可调整权重）
            anomaly_scores = 0.7 * mse1_norm + 0.3 * mse2_norm

        # 转换为numpy并处理极端值
        anomaly_scores = anomaly_scores.cpu().numpy()
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)

        # 确保分数非负（异常分数不能为负）
        anomaly_scores = np.maximum(anomaly_scores, 0.0)

        return anomaly_scores

    def evaluate(self, test_windows, test_labels, verbose=True):
        """快速评估模型性能"""
        anomaly_scores = self.predict(test_windows)

        # 计算核心指标
        auroc = roc_auc_score(test_labels, anomaly_scores)
        precision, recall, _ = precision_recall_curve(test_labels, anomaly_scores)
        aupr = auc(recall, precision)

        results = {
            "auroc": auroc,
            "aupr": aupr,
            "anomaly_scores": anomaly_scores
        }

        if verbose:
            print(f"\n📊 USAD评估结果：")
            print(f"   AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

        return results


# 测试代码
if __name__ == "__main__":
    # 模拟时序窗口数据
    np.random.seed(42)
    train_windows = np.random.randn(1000, 20, 5).astype(np.float32)
    test_windows = np.random.randn(200, 20, 5).astype(np.float32)
    # 模拟标签：10%异常
    test_labels = np.zeros(200)
    test_labels[np.random.choice(200, 20, replace=False)] = 1

    # 初始化检测器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = USADAnomalyDetector(input_dim=5, latent_dim=32, device=device)

    # 训练（传入训练标签，过滤异常）
    detector.train(train_windows, train_labels=np.zeros(1000), epochs=10, batch_size=16)

    # 评估
    detector.evaluate(test_windows, test_labels)