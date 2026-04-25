import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Conv-TasNet 中的时序卷积残差块（简化版）：
    1x1 卷积降维 + 深度可分离扩张卷积 + 残差连接
    """

    def __init__(self, channels, hidden_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation,
                groups=hidden_channels,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvTasNet_SEP(nn.Module):
    """
    适配当前工程的 Conv-TasNet 风格端到端分离网络。
    输入:  [B, n_mic, T]
    输出:  [B, n_source, T]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        seq_len=1024,
        enc_dim=128,
        bottleneck=128,
        hidden=256,
        kernel_size=3,
        num_blocks=4,
        num_repeats=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels

        # 编码器：将多通道混合波形映射到潜在特征
        self.encoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=enc_dim,
            kernel_size=16,
            stride=8,
            padding=8,
            bias=False,
        )

        # 分离器前端
        self.norm = nn.GroupNorm(1, enc_dim, eps=1e-8)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck, kernel_size=1)

        # TCN 主干
        tcn_blocks = []
        for _ in range(num_repeats):
            for b in range(num_blocks):
                tcn_blocks.append(
                    TemporalBlock(
                        channels=bottleneck,
                        hidden_channels=hidden,
                        kernel_size=kernel_size,
                        dilation=2**b,
                        dropout=dropout,
                    )
                )
        self.tcn = nn.Sequential(*tcn_blocks)

        # 估计每个源的掩码
        self.mask = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck, out_channels * enc_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 解码器：将每个源的潜在特征还原到时域
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_dim,
            out_channels=1,
            kernel_size=16,
            stride=8,
            padding=8,
            bias=False,
        )

    def forward(self, x):
        # [B, n_mic, T] -> [B, N, L]
        enc = self.encoder(x)
        feat = self.bottleneck(self.norm(enc))
        feat = self.tcn(feat)

        # [B, C*N, L] -> [B, C, N, L]
        bsz, _, frames = enc.shape
        mask = self.mask(feat).view(bsz, self.out_channels, -1, frames)

        # [B, C, N, L]
        masked = mask * enc.unsqueeze(1)

        # 每个源分别解码
        masked = masked.reshape(bsz * self.out_channels, -1, frames)
        decoded = self.decoder(masked)  # [B*C, 1, T']
        decoded = decoded.squeeze(1).reshape(bsz, self.out_channels, -1)

        # 与目标长度对齐
        if decoded.shape[-1] > self.seq_len:
            decoded = decoded[:, :, : self.seq_len]
        elif decoded.shape[-1] < self.seq_len:
            pad_len = self.seq_len - decoded.shape[-1]
            decoded = nn.functional.pad(decoded, (0, pad_len))
        return decoded
