import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset

from Contrast_Model.esn_model import train_windows, test_windows, test_labels


class LightweightLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 轻量化LSTM（仅1层，隐藏维度64）
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, input_dim)  # 重构输入

    def forward(self, x):
        # x: (batch_size, window_length, input_dim)
        lstm_out, _ = self.lstm(x)
        recon = self.fc(lstm_out)  # (batch_size, window_length, input_dim)
        return recon


# LSTM训练和预测
class LSTMAnomalyDetector:
    def __init__(self, input_dim, device="cpu"):
        self.device = device
        self.model = LightweightLSTM(input_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, train_windows, epochs=50, batch_size=16):
        train_dataset = TensorDataset(torch.FloatTensor(train_windows))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                x = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(x)
                loss = self.criterion(recon, x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")

    def predict(self, test_windows):
        self.model.eval()
        test_tensor = torch.FloatTensor(test_windows).to(self.device)
        with torch.no_grad():
            recon = self.model(test_tensor)
            # 计算每个窗口的MSE作为异常分数
            mse = torch.mean((recon - test_tensor) ** 2, dim=(1, 2))
        return mse.cpu().numpy()


# LSTM使用示例
input_dim = train_windows.shape[-1]
lstm_detector = LSTMAnomalyDetector(input_dim, device="cuda" if torch.cuda.is_available() else "cpu")
lstm_detector.train(train_windows, epochs=50)
lstm_scores = lstm_detector.predict(test_windows)

# 评估
auroc = roc_auc_score(test_labels, lstm_scores)
precision, recall, _ = precision_recall_curve(test_labels, lstm_scores)
aupr = auc(recall, precision)
print(f"LSTM - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")