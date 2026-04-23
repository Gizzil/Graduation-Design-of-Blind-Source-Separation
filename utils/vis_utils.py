import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_signal_comparison(source_signals, mixed_signals, separated_signals, fs, save_path):
    """
    绘制源信号、混合信号、分离信号对比图
    :param source_signals: 真实源信号，形状 [n_source, seq_len]
    :param mixed_signals: 混合信号，形状 [n_mic, seq_len]
    :param separated_signals: 各算法分离信号，字典 {算法名: 信号矩阵}
    :param fs: 采样率
    :param save_path: 保存路径
    """
    n_source = source_signals.shape[0]
    n_alg = len(separated_signals)
    total_rows = 2 + n_alg  # 源信号 + 混合信号 + 各算法分离信号
    seq_len = source_signals.shape[1]
    t = np.linspace(0, seq_len/fs, seq_len) * 1000  # 时间轴（ms）

    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2*total_rows), dpi=300)
    fig.subplots_adjust(hspace=0.5)

    # 绘制源信号
    for src_idx in range(n_source):
        axes[0].plot(t, source_signals[src_idx], label=f"源信号{src_idx+1}", linewidth=0.8)
    axes[0].set_title("原始源信号", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("时间 (ms)", fontsize=10)
    axes[0].set_ylabel("幅度", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # 绘制混合信号
    for mic_idx in range(mixed_signals.shape[0]):
        axes[1].plot(t, mixed_signals[mic_idx], label=f"混合信号{mic_idx+1}", linewidth=0.8)
    axes[1].set_title("观测混合信号", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("时间 (ms)", fontsize=10)
    axes[1].set_ylabel("幅度", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    # 绘制各算法分离信号
    for alg_idx, (alg_name, sep_signals) in enumerate(separated_signals.items()):
        ax = axes[2 + alg_idx]
        for src_idx in range(n_source):
            ax.plot(t, sep_signals[src_idx], label=f"分离信号{src_idx+1}", linewidth=0.8)
        ax.set_title(f"{alg_name} 分离结果", fontsize=12, fontweight='bold')
        ax.set_xlabel("时间 (ms)", fontsize=10)
        ax.set_ylabel("幅度", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_time_freq(signals, fs, channel_labels, suptitle, save_path):
    n_channel, seq_len = signals.shape
    t = np.arange(seq_len)
    freqs = np.fft.rfftfreq(seq_len, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(signals, axis=1))
    fig, axes = plt.subplots(2, n_channel, figsize=(4 * n_channel, 4), dpi=300)
    if n_channel == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for idx in range(n_channel):
        axes[0, idx].plot(t, signals[idx], linewidth=0.8)
        axes[0, idx].set_title(channel_labels[idx], fontsize=10)
        axes[0, idx].set_xlabel("采样点", fontsize=9)
        axes[0, idx].set_ylabel("幅度", fontsize=9)
        axes[0, idx].grid(alpha=0.3)
        axes[1, idx].plot(freqs, spec[idx], linewidth=0.8)
        axes[1, idx].set_xlabel("频率 (Hz)", fontsize=9)
        axes[1, idx].set_ylabel("幅度", fontsize=9)
        axes[1, idx].grid(alpha=0.3)
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_robustness_curve(robustness_df, save_path):
    """
    绘制不同SNR下各算法的SDR变化曲线
    :param robustness_df: 包含 SNR(dB) 与各算法 SDR 的 DataFrame
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 6), dpi=300)
    snr = robustness_df["SNR(dB)"].tolist()
    algorithms = [col for col in robustness_df.columns if col != "SNR(dB)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, alg in enumerate(algorithms):
        plt.plot(snr, robustness_df[alg], marker="o", linewidth=2, color=colors[idx % len(colors)], label=alg)
    plt.title("不同SNR下各算法SDR随噪声强度的变化", fontsize=14, fontweight="bold")
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("平均SDR (dB)", fontsize=12)
    plt.xticks(snr)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_loss, val_loss, save_path):
    """
    绘制训练/验证损失曲线
    :param train_loss: 训练损失列表
    :param val_loss: 验证损失列表
    :param save_path: 保存路径
    """
    epochs = np.arange(1, len(train_loss)+1)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_loss, label="训练损失", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_loss, label="验证损失", color="#ff7f0e", linewidth=2)
    plt.title("模型训练损失曲线", fontsize=14, fontweight='bold')
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("MSE损失", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_df, save_path):
    alg_mean = metrics_df.groupby("算法").mean(numeric_only=True)
    algorithms = alg_mean.index.tolist()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    axes[0].bar(algorithms, alg_mean["SDR(dB)"], color=colors, width=0.6)
    axes[0].set_title("各算法平均SDR对比", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("SDR (dB)", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(algorithms, alg_mean["MSE"], color=colors, width=0.6)
    axes[1].set_title("各算法平均MSE对比", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MSE", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(algorithms, alg_mean["相关系数"], color=colors, width=0.6)
    axes[2].set_title("各算法平均相关系数对比", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("相关系数", fontsize=10)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis="y", alpha=0.3)

    if "全局拒绝水平(GRL)" in alg_mean.columns:
        axes[3].bar(algorithms, alg_mean["全局拒绝水平(GRL)"], color=colors, width=0.6)
        axes[3].set_title("各算法全局拒绝水平对比", fontsize=12, fontweight="bold")
        axes[3].set_ylabel("GRL", fontsize=10)
        axes[3].grid(axis="y", alpha=0.3)
    else:
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
