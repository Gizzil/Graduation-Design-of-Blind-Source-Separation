import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """
    TCN基本块：包含两个扩张卷积层和残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,  # 保持长度不变
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接：如果输入输出通道不同，使用1x1卷积调整
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.bn2(self.conv2(x)))
        return self.relu2(x + residual)

class TCN_SEP(nn.Module):
    """
    时序卷积网络 (TCN) 端到端盲源分离模型
    使用扩张卷积捕捉长期时间依赖，适合时序信号分离
    """
    def __init__(self, in_channels=3, out_channels=3, seq_len=1024, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_SEP, self).__init__()
        self.seq_len = seq_len # 尽管在新设计中不再需要，但保留以兼容其他代码
        self.num_levels = len(num_channels)

        # 编码器：堆叠TCN块，逐步增加通道数和扩张率
        layers = []
        for i in range(self.num_levels):
            dilation = 2 ** i  # 扩张率：1, 2, 4, ...
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn_encoder = nn.Sequential(*layers)

        # 输出层：使用1x1卷积将特征映射到输出通道
        self.output_conv = nn.Conv1d(num_channels[-1], out_channels, 1)

    def forward(self, x):
        """
        前向传播
        :param x: 混合信号，形状 [batch_size, in_channels, seq_len]
        :return: 分离后的源信号，形状 [batch_size, out_channels, seq_len]
        """
        # TCN编码器处理后，序列长度不变
        features = self.tcn_encoder(x)
        
        # 通过1x1卷积得到最终输出
        output = self.output_conv(features)
        
        return output