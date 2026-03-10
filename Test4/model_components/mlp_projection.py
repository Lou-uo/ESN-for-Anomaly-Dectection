import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjection(nn.Module):
    """MLP投影头 + 几何归一化（单位超球面投射）+ 层归一化"""
    def __init__(self, input_dim=500, mid_dim=256, emb_dim=128):
        super(MLPProjection, self).__init__()
        self.mlp1 = nn.Linear(input_dim, mid_dim)
        self.mlp2 = nn.Linear(mid_dim, emb_dim)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(mid_dim)  # 新增层归一化
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        h = self.relu(self.ln1(self.mlp1(x)))
        z_raw = self.ln2(self.mlp2(h))  # (batch_size, emb_dim)
        # 关键：添加1e-8防止模长为0
        r = torch.norm(z_raw, p=2, dim=1, keepdim=True) + 1e-8
        alpha = z_raw / r  # (batch_size, emb_dim)
        return alpha, r