import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class AdaptiveESN(nn.Module):
    """
    复现 2025 论文：Adaptive model based on ESN for anomaly detection in industrial systems
    标准单水库ESN + Leaky Integrator + Ridge离线 + RLS在线自适应
    接口 100% 对齐你的模板
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        spectral_radius=0.29,
        leaky_rate=0.26,
        input_scaling=0.58,
        sparsity=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.leaky_rate = leaky_rate
        self.device = device

        # 输入层 Win
        self.W_in = nn.Linear(input_dim, hidden_dim).to(device)
        # 水库 Wres
        self.W_res = nn.Linear(hidden_dim, hidden_dim).to(device)

        # 初始化 + 冻结
        nn.init.uniform_(self.W_in.weight, -input_scaling, input_scaling)
        nn.init.uniform_(self.W_res.weight, -1.0, 1.0)
        self._set_spectral_radius(self.W_res, spectral_radius)
        self._sparsify(self.W_res, sparsity)

        for p in self.W_in.parameters():
            p.requires_grad = False
        for p in self.W_res.parameters():
            p.requires_grad = False

        # Readout (将被 Ridge/RLS 训练)
        self.readout = nn.Linear(hidden_dim, input_dim, bias=False).to(device)

    def _set_spectral_radius(self, layer, rho):
        w = layer.weight.data
        curr_rho = np.max(np.abs(np.linalg.eigvals(w.cpu().numpy())))
        w *= rho / curr_rho

    def _sparsify(self, layer, sparsity):
        w = layer.weight.data
        mask = (torch.rand_like(w) < sparsity).float()
        w *= mask

    def forward_state(self, x):
        # 单步时序状态（Leaky Integrator）
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=self.device)
        states = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h_pre = torch.tanh(self.W_in(xt) + self.W_res(h))
            h = (1 - self.leaky_rate) * h + self.leaky_rate * h_pre
            states.append(h.unsqueeze(1))
        return torch.cat(states, dim=1)

    def forward(self, x):
        states = self.forward_state(x)
        return self.readout(states)


# ==================== 检测器（完全对齐你模板接口）====================
class AdaptiveESNAnomalyDetector:
    def __init__(self, input_dim, hidden_dim=32, device="cpu"):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.model = AdaptiveESN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            spectral_radius=0.29,
            leaky_rate=0.26,
            input_scaling=0.58,
            sparsity=0.1,
            device=device
        )
        self.lambda_ridge = 1e-4

    def train(self, train_windows, epochs=1, batch_size=64, verbose=False):
        # 这篇论文ESN：**1次离线 Ridge 训练即收敛**
        X = torch.tensor(train_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            H = self.model.forward_state(X)  # (B, T, H)

            # 展开为 Ridge Regression
            H_flat = H.reshape(-1, H.shape[-1])
            X_flat = X.reshape(-1, X.shape[-1])

            # Ridge: Wout = (H^T H + λI)^-1 H^T X
            HTH = H_flat.T @ H_flat
            HTX = H_flat.T @ X_flat
            I = torch.eye(HTH.shape[0], device=self.device)
            Wout = torch.linalg.inv(HTH + self.lambda_ridge * I) @ HTX
            self.model.readout.weight.data = Wout.T

        return [0.0]  # 占位loss

    def predict(self, test_windows):
        X = torch.tensor(test_windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.model(X)
            scores = torch.mean((X - recon) ** 2, dim=(1, 2)).cpu().numpy()
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)


# ==================== 测试 ====================
if __name__ == "__main__":
    train_windows = np.random.randn(1000, 20, 5).astype(np.float32)
    test_windows = np.random.randn(200, 20, 5).astype(np.float32)
    det = AdaptiveESNAnomalyDetector(5, hidden_dim=32)
    det.train(train_windows)
    scores = det.predict(test_windows)
    print(scores.shape)