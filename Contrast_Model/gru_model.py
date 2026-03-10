import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')


class LightweightGRU(nn.Module):
    """轻量化GRU重构模型（适配时序异常检测）"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, *args, **kwargs):
        gru_out, _ = self.gru(x)
        recon = self.fc(gru_out)
        return recon


# 修复后的train_model函数
def train_model(model, train_loader, criterion, optimizer, epochs, device, clip_norm=0.5, verbose=True):
    """
    训练模型（修复NaN/Inf判断和损失标量化问题）
    """
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)

            # 1. 确保损失是标量（核心）
            loss = criterion(output, batch_y)

            # 2. 安全判断NaN/Inf（修复布尔值歧义错误）
            loss_val = loss.item()  # 转为Python浮点数
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"⚠️ Epoch {epoch + 1} 检测到无效损失值 {loss_val}，跳过本轮")
                continue

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            epoch_loss += loss_val * batch_x.size(0)

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # 打印日志
        if verbose and (epoch + 1) % 10 == 0:
            print(f"训练 | Epoch {epoch + 1}/{epochs} | 平均MSE损失: {avg_loss:.6f}")

    return model, loss_history


# 整合后的GRU检测器（调用修复后的train_model）
class GRUAnomalyDetector:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, device="cpu"):
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.model = LightweightGRU(input_dim, hidden_dim, num_layers).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')  # 确保损失是标量
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-6)

    def train(self, train_windows, epochs=50, batch_size=16, verbose=True):
        # 构建DataLoader
        train_tensor = torch.tensor(train_windows, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 调用修复后的train_model
        self.model, loss_history = train_model(
            model=self.model,
            train_loader=train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            epochs=epochs,
            device=self.device,
            verbose=verbose
        )
        return loss_history

    def predict(self, test_windows):
        self.model.eval()
        test_tensor = torch.tensor(test_windows, dtype=torch.float32).to(self.device)
        anomaly_scores = []

        with torch.no_grad():
            for i in range(0, len(test_tensor), 32):
                batch = test_tensor[i:i + 32]
                recon = self.model(batch)
                batch_mse = torch.mean((batch - recon) ** 2, dim=(1, 2)).cpu().numpy()
                anomaly_scores.extend(batch_mse)

        anomaly_scores = np.array(anomaly_scores)
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return anomaly_scores


# ========== 测试运行 ==========
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    train_windows = np.random.randn(1000, 20, 5).astype(np.float32)
    test_windows = np.random.randn(200, 20, 5).astype(np.float32)
    test_labels = np.random.randint(0, 2, size=200)

    # 初始化检测器
    detector = GRUAnomalyDetector(
        input_dim=5,
        hidden_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 训练（调用修复后的train_model）
    loss_history = detector.train(train_windows, epochs=10, batch_size=16)

    # 预测和评估
    scores = detector.predict(test_windows)
    auroc = roc_auc_score(test_labels, scores)
    precision, recall, _ = precision_recall_curve(test_labels, scores)
    aupr = auc(recall, precision)

    print(f"\n异常检测结果：")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")