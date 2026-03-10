import torch
import torch.nn as nn
import torch.nn.functional as F

class Reservoir(nn.Module):
    """回声状态网络储备池层（优化：增加dropout+批量归一化）"""
    def __init__(self, input_dim, hidden_dim=500, spectral_radius=0.9, leaking_rate=0.05, sparsity=0.1, dropout=0.1):
        super(Reservoir, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.sparsity = sparsity
        self.dropout = nn.Dropout(dropout)  # 新增dropout

        # 优化1：更稳定的权重初始化
        self.W_in = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.001, requires_grad=False)
        self.W_res = self._initialize_reservoir_weights()
        # 优化2：批量归一化稳定隐藏状态
        self.bn = nn.BatchNorm1d(hidden_dim)

    def _initialize_reservoir_weights(self):
        W = torch.randn(self.hidden_dim, self.hidden_dim) * 0.001
        mask = torch.rand_like(W) < self.sparsity
        W[mask] = 0
        # 处理空特征值（避免除零）
        eigvals = torch.linalg.eigvals(W).real
        max_eig = torch.max(torch.abs(eigvals)) if len(eigvals) > 0 else 1.0
        if max_eig == 0:
            max_eig = 1.0
        W = W / max_eig * self.spectral_radius
        return nn.Parameter(W, requires_grad=False)

    def forward(self, x):
        batch_size, window_length, _ = x.shape
        hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(window_length):
            x_t = x[:, t, :]
            linear_term = torch.matmul(x_t, self.W_in.T) + torch.matmul(hidden_state, self.W_res.T)
            # 限制tanh输入范围（防止溢出）
            linear_term = torch.clamp(linear_term, min=-10, max=10)
            hidden_state = (1 - self.leaking_rate) * hidden_state + self.leaking_rate * torch.tanh(linear_term)
            # 过滤NaN/Inf + dropout + BN
            hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=1e3, neginf=-1e3)
            hidden_state = self.dropout(hidden_state)
            hidden_state = self.bn(hidden_state)

        return hidden_state