import torch
import torch.nn as nn

class MLP_SEP(nn.Module):
    """
    轻量MLP端到端盲源分离模型
    适用于低维度信号快速训练与基线对比
    """
    def __init__(self, in_dim=2, out_dim=2, seq_len=1024):
        super(MLP_SEP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * seq_len, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim * seq_len)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 混合信号，形状 [batch_size, in_channels, seq_len]
        :return: 分离后的源信号，形状 [batch_size, out_channels, seq_len]
        """
        batch_size = x.shape[0]
        # 展平
        x_flat = x.reshape(batch_size, -1)
        # MLP前向
        out_flat = self.mlp(x_flat)
        # 恢复形状
        out = out_flat.reshape(batch_size, self.out_dim, self.seq_len)
        return out