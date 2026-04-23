import torch
import torch.nn as nn

class CNN_SEP(nn.Module):
    """
    1D-CNN编码器-解码器 端到端盲源分离模型
    适配时序信号分离，修复输出长度不匹配问题：输入1024 → 输出1024
    """
    def __init__(self, in_channels=2, out_channels=2, seq_len=1024):
        super(CNN_SEP, self).__init__()
        self.seq_len = seq_len

        # 编码器：提取混合信号时序特征（输入1024 → 输出128）
        self.encoder = nn.Sequential(
            # 第一层卷积：1024 → 512 (kernel=8, stride=2, padding=3)
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=8,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 第二层卷积：512 → 256
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=8,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # 第三层卷积：256 → 128
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=8,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 解码器：还原源信号（输入128 → 输出1024）
        self.decoder = nn.Sequential(
            # 第一层转置卷积：128 → 256
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=8,
                stride=2,
                padding=3,
                output_padding=0  # 关键修正：取消output_padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # 第二层转置卷积：256 → 512
            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=64,
                kernel_size=8,
                stride=2,
                padding=3,
                output_padding=0
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 第三层转置卷积：512 → 1024
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=8,
                stride=2,
                padding=3,
                output_padding=0
            )
        )

    def forward(self, x):
        """
        前向传播
        :param x: 混合信号，形状 [batch_size, in_channels, seq_len]
        :return: 分离后的源信号，形状 [batch_size, out_channels, seq_len]
        """
        x = self.encoder(x)
        x = self.decoder(x)
        # 最终校验：如果长度仍有偏差，强制截断/补零到目标长度（兜底方案）
        if x.shape[-1] != self.seq_len:
            x = x[:, :, :self.seq_len]  # 截断过长部分
            # 补零（如果太短）
            pad_len = self.seq_len - x.shape[-1]
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, pad_len))
        return x