import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class LightweightESN_AE(nn.Module):
    """De Vita 2023 ESN-AE 稳定快速版"""
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        code_dim=16,
        spectral_radius=0.9,
        sparsity=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.device = device

        # ===================== Encoder Reservoir (fixed) =====================
        self.W_in_enc = nn.Linear(input_dim, hidden_dim).to(device)
        self.W_res_enc = nn.Linear(hidden_dim, hidden_dim).to(device)

        nn.init.uniform_(self.W_in_enc.weight, -1.0, 1.0)
        nn.init.uniform_(self.W_res_enc.weight, -1.0, 1.0)
        self._set_spectral_radius(self.W_res_enc, spectral_radius)
        self._sparsify(self.W_res_enc, sparsity)

        for p in self.W_in_enc.parameters():
            p.requires_grad = False
        for p in self.W_res_enc.parameters():
            p.requires_grad = False

        # ===================== Code Layer (trainable) =====================
        self.code = nn.Linear(hidden_dim, code_dim)
        self.code_out = nn.Linear(code_dim, hidden_dim)

        # ===================== Decoder Reservoir (fixed) =====================
        self.W_res_dec = nn.Linear(hidden_dim, hidden_dim).to(device)
        nn.init.uniform_(self.W_res_dec.weight, -1.0, 1.0)
        self._set_spectral_radius(self.W_res_dec, spectral_radius)
        self._sparsify(self.W_res_dec, sparsity)

        for p in self.W_res_dec.parameters():
            p.requires_grad = False

        # Readout
        self.readout = nn.Linear(hidden_dim, input_dim)

    def _set_spectral_radius(self, layer, rho):
        w = layer.weight.data
        curr_rho = np.max(np.abs(np.linalg.eigvals(w.cpu().numpy())))
        w *= rho / curr_rho

    def _sparsify(self, layer, sparsity):
        w = layer.weight.data
        mask = (torch.rand_like(w) < sparsity).float()
        w *= mask

    def forward(self, x):
        B, seq_len, _ = x.shape
        device = x.device

        # ===================== Encoder =====================
        h_enc = torch.zeros(B, self.hidden_dim, device=device)
        enc_out = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h_enc = torch.tanh(self.W_in_enc(xt) + self.W_res_enc(h_enc))
            enc_out.append(h_enc.unsqueeze(1))
        enc_out = torch.cat(enc_out, dim=1)

        # ===================== Code =====================
        z = self.code(enc_out)
        h_dec_in = self.code_out(z)

        # ===================== Decoder =====================
        h_dec = torch.zeros(B, self.hidden_dim, device=device)
        dec_out = []
        for t in range(seq_len):
            ht = h_dec_in[:, t, :]
            h_dec = torch.tanh(ht + self.W_res_dec(h_dec))
            dec_out.append(h_dec.unsqueeze(1))
        dec_out = torch.cat(dec_out, dim=1)

        recon = self.readout(dec_out)
        return recon


# ===================== 训练函数（对齐你的模板） =====================
def train_model(model, train_loader, criterion, optimizer, epochs, device, clip_norm=0.5, verbose=True):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f}")
    return model, loss_history


# ===================== 检测器（完全对齐你的接口） =====================
class ESNAnomalyDetector:
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, device="cpu"):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.model = LightweightESN_AE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            code_dim=hidden_dim//2,
            device=device
        ).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=1e-6
        )

    def train(self, train_windows, epochs=20, batch_size=32, verbose=False):
        train_tensor = torch.tensor(train_windows, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
        self.model, loss_history = train_model(
            self.model, loader, self.criterion, self.optimizer, epochs, self.device, verbose=verbose
        )
        return loss_history

    def predict(self, test_windows):
        self.model.eval()
        test_tensor = torch.tensor(test_windows, dtype=torch.float32, device=self.device)
        scores = []
        with torch.no_grad():
            for i in range(0, len(test_tensor), 64):
                batch = test_tensor[i:i+64]
                recon = self.model(batch)
                mse = torch.mean((batch - recon)**2, dim=(1,2)).cpu().numpy()
                scores.extend(mse)
        return np.nan_to_num(np.array(scores), nan=0.0, posinf=1e6, neginf=-1e6)