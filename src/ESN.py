import torch
from torch import nn


# 修正 ESN 初始化中的数据类型
class ESN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, spectral_radius=0.9):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. 让所有计算保持 float32
        w = torch.rand(hidden_size, hidden_size, dtype=torch.float32) - 0.5
        eigenvalues, _ = torch.linalg.eig(
            w.to(torch.float64)
        )  # 只在求特征值时用 float64
        max_eig = eigenvalues.abs().max().float()  # 取 float32，避免 float64 残留

        # 2. 初始化权重并确保所有都是 float32
        self.W_res = nn.Parameter(
            (w * (spectral_radius / max_eig)).float(), requires_grad=False
        )
        self.W_in = nn.Parameter(
            torch.rand(hidden_size, input_size, dtype=torch.float32) - 0.5,
            requires_grad=False,
        )
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(
            batch_size, self.hidden_size, dtype=torch.float32, device=x.device
        )  # 让 h 也用 float32
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            h = torch.tanh(
                torch.matmul(h, self.W_res.t()) + torch.matmul(u, self.W_in.t())
            )
            out = self.W_out(h)
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)
