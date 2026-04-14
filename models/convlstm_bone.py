# Arsitektur ConvLSTM untuk prediksi sisa usia simpan makanan.
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Satu sel ConvLSTM. Operasi matrix multiply diganti Conv2d
    agar struktur spasial [H, W] dari feature map tetap terjaga.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,          
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )
        nn.init.constant_(
            self.conv.bias[hidden_channels : 2 * hidden_channels], 1.0
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        x : [B, in_channels, H, W]   — input frame saat ini
        h : [B, hidden_channels, H, W] — hidden state sebelumnya
        c : [B, hidden_channels, H, W] — cell state sebelumnya
        """
        gates       = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, B: int, H: int, W: int, device: torch.device):
        """Inisialisasi hidden & cell state dengan zeros."""
        z = lambda: torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z(), z()


class ShelfLifePredictor(nn.Module):
    """
    Stack 2 layer ConvLSTM + regression head.

    Input  : [B, T, C, H, W] — sekuens feature maps dari backbone OSR
    Output : [B, 1]           — prediksi hari tersisa (float, non-negatif)

    Ukuran model sengaja kecil (hidden1=16, hidden2=8, ~15K params)
    agar tidak overfit pada dataset kecil.
    """
    def __init__(self, in_channels: int, hidden1: int = 16, hidden2: int = 8):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_channels, hidden1)
        self.cell2 = ConvLSTMCell(hidden1,     hidden2)

        self.bn = nn.BatchNorm2d(hidden2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),   
            nn.Flatten(),                  
            nn.Linear(hidden2 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Softplus()                 
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, C, H, W]
        """
        B, T, C, H, W = seq.shape

        h1, c1 = self.cell1.init_hidden(B, H, W, seq.device)
        h2, c2 = self.cell2.init_hidden(B, H, W, seq.device)

        for t in range(T):
            h1, c1 = self.cell1(seq[:, t], h1, c1)
            h2, c2 = self.cell2(h1,        h2, c2)

        h2 = self.bn(h2)
        return self.head(h2)   