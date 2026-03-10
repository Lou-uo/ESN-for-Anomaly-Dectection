import torch
import torch.nn as nn
import torch.nn.functional as F

class MCCLoss(nn.Module):
    """最大相关熵损失 + Focal Loss融合（平衡精准度-召回率）"""
    def __init__(self, sigma=1.0, lambda1=1e-5, lambda2=1e-6, gamma=2.0):
        super(MCCLoss, self).__init__()
        self.sigma = sigma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma  # Focal Loss gamma

    def forward(self, y_pred, y_target, W_out, mlp_params):
        # 1. 计算误差（绝对误差，避免正负抵消）
        error = torch.abs(y_pred - y_target)
        error_norm = torch.norm(error, p=2, dim=1) + 1e-8

        # 2. 高斯核（限制数值范围，防止exp溢出）
        kernel_arg = - (error_norm / self.sigma) ** 2 / 2
        kernel_arg = torch.clamp(kernel_arg, min=-50, max=50)
        kernel_weight = torch.exp(kernel_arg)

        # 3. 核心损失 + Focal Loss（聚焦难样本）
        core_loss = torch.mean((1 - kernel_weight) * torch.pow(1 - kernel_weight, self.gamma - 1))

        # 4. 正则化项（clip防止梯度爆炸）
        reg1 = self.lambda1 * torch.clamp(torch.norm(W_out.weight, p=2) ** 2, 0, 1e3)
        reg2 = self.lambda2 * sum(torch.clamp(torch.norm(param, p=2) ** 2, 0, 1e3) for param in mlp_params)

        # 5. 总损失clip
        total_loss = core_loss + reg1 + reg2
        total_loss = torch.clamp(total_loss, min=0, max=1e3)
        return total_loss