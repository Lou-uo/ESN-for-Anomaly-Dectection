import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """可逆实例归一化（Reversible Instance Normalization）"""
    def __init__(self, n_vars, eps=1e-5):
        super(RevIN, self).__init__()
        self.n_vars = n_vars  # 变量数
        self.eps = eps
        # 可学习仿射参数
        self.gamma = nn.Parameter(torch.ones(1, 1, n_vars))  # (1,1,N_var)
        self.beta = nn.Parameter(torch.zeros(1, 1, n_vars))   # (1,1,N_var)

    def forward(self, x, mode="normalize"):
        """
        Args:
            x: 输入张量 (batch_size, window_length, n_vars)
            mode: "normalize"（归一化） / "denormalize"（反归一化）
        Returns:
            处理后张量
        """
        if mode == "normalize":
            # 1. 计算每个窗口的均值和标准差（按变量维度）
            self.mu = torch.mean(x, dim=1, keepdim=True)  # (batch_size,1,n_vars)
            self.sigma = torch.std(x, dim=1, keepdim=True) + self.eps  # (batch_size,1,n_vars)
            # 2. 归一化
            x_norm = (x - self.mu) / self.sigma
            # 3. 仿射变换
            x_norm = x_norm * self.gamma + self.beta
            return x_norm
        elif mode == "denormalize":
            # 1. 逆仿射变换
            x_denorm = (x - self.beta) / self.gamma
            # 2. 逆统计归一化
            x_denorm = x_denorm * self.sigma + self.mu
            return x_denorm
        else:
            raise ValueError("mode must be 'normalize' or 'denormalize'")