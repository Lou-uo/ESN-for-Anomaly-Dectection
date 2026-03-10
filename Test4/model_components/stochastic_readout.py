import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticReadout(nn.Module):
    """随机读出层（DropConnect + 权重归一化）"""

    def __init__(self, emb_dim=128, output_dim=3, mask_rate=0.1):
        super(StochasticReadout, self).__init__()
        self.emb_dim = emb_dim  # 嵌入维度
        self.output_dim = output_dim  # 输出维度（n_vars）
        self.mask_rate = mask_rate  # 权重丢弃率
        # 读出权重 + 权重归一化
        self.W_out = nn.Linear(emb_dim, output_dim, bias=False)
        nn.init.kaiming_normal_(self.W_out.weight)  # 优化初始化

    def forward(self, alpha, training=True):
        """
        Args:
            alpha: 几何归一化嵌入 (batch_size, emb_dim)
            training: 是否训练模式（训练时启用DropConnect）
        Returns:
            y_pred: 预测输出 (batch_size, output_dim)
        """
        # 权重归一化（提升稳定性）
        W_norm = F.normalize(self.W_out.weight, p=2, dim=1)

        if training:
            # 生成Bernoulli掩码（1-保留，0-丢弃）
            mask = torch.bernoulli(torch.ones_like(W_norm) * (1 - self.mask_rate)).to(alpha.device)
            # DropConnect：权重元素级相乘
            W_out_masked = W_norm * mask
            y_pred = torch.matmul(alpha, W_out_masked.T)
        else:
            # 推理时不使用掩码
            y_pred = torch.matmul(alpha, W_norm.T)
        return y_pred