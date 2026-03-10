import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components.rev_in import RevIN
from model_components.reservoir import Reservoir
from model_components.mlp_projection import MLPProjection
from model_components.stochastic_readout import StochasticReadout
from model_components.loss import MCCLoss
# 新增：导入动态阈值类（如果train_eval和model在同一目录）
from train_eval import POTDynamicThreshold

class CESNAnomalyDetector(nn.Module):
    """基于CESN的多变量时间序列异常检测器（SOTA优化版）"""
    def __init__(self, n_vars=3, window_length=50, hidden_dim=500, emb_dim=128,
                 spectral_radius=0.9, leaking_rate=0.05, mask_rate=0.1,
                 sigma=5.0, lambda1=1e-5, lambda2=1e-6):
        super(CESNAnomalyDetector, self).__init__()
        self.n_vars = n_vars
        self.window_length = window_length

        self.rev_in = RevIN(n_vars=n_vars)
        self.reservoir = Reservoir(
            input_dim=n_vars,
            hidden_dim=hidden_dim,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            dropout=0.1
        )
        self.mlp_proj = MLPProjection(
            input_dim=hidden_dim,
            mid_dim=hidden_dim // 2,
            emb_dim=emb_dim
        )
        self.stochastic_readout = StochasticReadout(
            emb_dim=emb_dim,
            output_dim=n_vars,
            mask_rate=mask_rate
        )
        self.mcc_loss = MCCLoss(
            sigma=sigma,
            lambda1=lambda1,
            lambda2=lambda2,
            gamma=2.0
        )
        # 新增：初始化动态阈值器（训练后校准）
        self.dynamic_thresholder = None

    def forward(self, x, y_target=None, training=True):
        x_norm = self.rev_in(x, mode="normalize")
        self.mu_cache = self.rev_in.mu
        self.sigma_cache = self.rev_in.sigma

        reservoir_state = self.reservoir(x_norm)
        reservoir_state = torch.nan_to_num(reservoir_state, nan=0.0, posinf=1e3, neginf=-1e3)

        alpha, r = self.mlp_proj(reservoir_state)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1e3, neginf=-1e3)

        y_pred = self.stochastic_readout(alpha, training=training)
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e3, neginf=-1e3)

        if training:
            y_target_norm = (y_target - self.mu_cache.squeeze(1)) / self.sigma_cache.squeeze(1)
            y_target_norm = y_target_norm * self.rev_in.gamma.squeeze(1) + self.rev_in.beta.squeeze(1)
            y_target_norm = torch.nan_to_num(y_target_norm, nan=0.0, posinf=1e3, neginf=-1e3)

            mlp_params = [self.mlp_proj.mlp1.weight, self.mlp_proj.mlp1.bias,
                          self.mlp_proj.mlp2.weight, self.mlp_proj.mlp2.bias]
            loss = self.mcc_loss(y_pred, y_target_norm, self.stochastic_readout.W_out, mlp_params)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)
            # # 消融MCC：仅用MSE损失（最基础的重建损失）
            # loss = F.mse_loss(y_pred, y_target_norm)
            return loss
        else:
            y_pred_denorm = self.rev_in(y_pred.unsqueeze(1), mode="denormalize").squeeze(1)
            return y_pred_denorm, r

    def predict_anomaly_score(self, x, K=10):
        self.eval()
        with torch.no_grad():
            x_norm = self.rev_in(x, mode="normalize")
            reservoir_state = self.reservoir(x_norm)
            alpha, r = self.mlp_proj(reservoir_state)

            y_preds = []
            for _ in range(K):
                y_pred = self.stochastic_readout(alpha, training=True)
                y_preds.append(y_pred)
            y_preds = torch.stack(y_preds, dim=0)
            pred_mean = torch.mean(y_preds, dim=0)

            pred_mean_denorm = self.rev_in(pred_mean.unsqueeze(1), mode="denormalize").squeeze(1)
            x_target = x[:, -1, :]

            recon_error = torch.abs(x_target - pred_mean_denorm)
            geo_dist = 1 - torch.sum(F.normalize(x_target, p=2, dim=1) * F.normalize(pred_mean_denorm, p=2, dim=1), dim=1)
            recon_score = torch.mean(recon_error, dim=1)
            anomaly_scores = 0.7 * recon_score + 0.3 * geo_dist

            anomaly_scores = anomaly_scores.cpu().numpy()
            if hasattr(self, 'train_error_mean'):
                anomaly_scores = (anomaly_scores - self.train_error_mean) / (self.train_error_std + 1e-8)
            else:
                anomaly_scores = (anomaly_scores - np.mean(anomaly_scores)) / (np.std(anomaly_scores) + 1e-8)

            anomaly_scores = np.clip(anomaly_scores, a_min=0.0, a_max=None)
            if hasattr(self, 'train_error_95q'):
                anomaly_scores = np.minimum(anomaly_scores, self.train_error_95q * 3)
            anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)

        return anomaly_scores

    def calibrate_train_stats(self, train_windows):
        self.eval()
        with torch.no_grad():
            train_errors = []
            train_terms1 = []
            train_terms2 = []
            train_terms3 = []
            train_r = []

            for i in range(0, len(train_windows), 32):
                batch = torch.tensor(train_windows[i: i + 32], dtype=torch.float32).to(next(self.parameters()).device)
                x_norm = self.rev_in(batch, mode="normalize")
                reservoir_state = self.reservoir(x_norm)
                alpha, r = self.mlp_proj(reservoir_state)

                y_pred = self.stochastic_readout(alpha, training=False)
                y_pred_denorm = self.rev_in(y_pred.unsqueeze(1), mode="denormalize").squeeze(1)
                x_target = batch[:, -1, :]
                recon_error = torch.mean(torch.abs(x_target - y_pred_denorm), dim=1)
                train_errors.extend(recon_error.cpu().numpy())

                train_r.extend(r.cpu().numpy())

                y_target = batch[:, -1, :]
                y_target_norm = (y_target - self.rev_in.mu.squeeze(1)) / self.rev_in.sigma.squeeze(1)
                y_target_norm = y_target_norm * self.rev_in.gamma.squeeze(1) + self.rev_in.beta.squeeze(1)

                y_preds = [self.stochastic_readout(alpha, training=True) for _ in range(5)]
                y_preds = torch.stack(y_preds, dim=0)
                pred_mean = torch.mean(y_preds, dim=0)

                target_var = torch.var(y_target_norm, dim=0) + 1e-8
                term1 = torch.mean((y_target_norm - pred_mean) ** 2 / target_var, dim=1)
                term2 = torch.mean(torch.var(y_preds, dim=0) + 1e-8, dim=1)

                r_mean_batch = torch.mean(r) + 1e-8
                term3 = (1 - r / r_mean_batch) ** 2

                train_terms1.extend(torch.nan_to_num(term1, nan=0.0).cpu().numpy())
                train_terms2.extend(torch.nan_to_num(term2, nan=0.0).cpu().numpy())
                train_terms3.extend(torch.nan_to_num(term3, nan=0.0).cpu().numpy())

            train_errors = np.array(train_errors) if train_errors else np.array([0.0])
            self.train_error_mean = np.mean(train_errors) + 1e-8
            self.train_error_std = np.std(train_errors) + 1e-8
            self.train_error_95q = np.percentile(train_errors, 95) + 1e-8

            train_terms1 = np.array(train_terms1) if train_terms1 else np.array([0.0])
            train_terms2 = np.array(train_terms2) if train_terms2 else np.array([0.0])
            train_terms3 = np.array(train_terms3) if train_terms3 else np.array([0.0])
            train_r = np.array(train_r) if train_r else np.array([1.0])

            self.train_r_mean = torch.tensor(np.mean(train_r) + 1e-8).float()
            self.train_term1_mean = torch.tensor(np.mean(train_terms1) + 1e-8).float()
            self.train_term1_std = torch.tensor(np.std(train_terms1) + 1e-8).float()
            self.train_term2_mean = torch.tensor(np.mean(train_terms2) + 1e-8).float()
            self.train_term2_std = torch.tensor(np.std(train_terms2) + 1e-8).float()
            self.train_term3_mean = torch.tensor(np.mean(train_terms3) + 1e-8).float()
            self.train_term3_std = torch.tensor(np.std(train_terms3) + 1e-8).float()

            # ========== 新增：用训练集正常样本预热动态阈值器 ==========
            self.dynamic_thresholder = POTDynamicThreshold(window_size=1000, init_quantile=0.98, risk_q=1e-3)
            for score in train_errors:
                self.dynamic_thresholder.update(score, is_anomaly=False)