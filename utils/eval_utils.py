import numpy as np

def calculate_mse(s_true, s_hat):
    """
    计算均方误差MSE
    :param s_true: 真实源信号，形状 [n_samples, n_sources, seq_len]
    :param s_hat: 估计源信号，形状同上
    :return: 每个样本每个源的MSE，形状 [n_samples, n_sources]
    """
    mse = np.mean((s_true - s_hat) ** 2, axis=-1)
    return mse

def calculate_sdr(s_true, s_hat):
    """
    计算信号失真比SDR（dB）
    :param s_true: 真实源信号，形状 [n_samples, n_sources, seq_len]
    :param s_hat: 估计源信号，形状同上
    :return: 每个样本每个源的SDR，形状 [n_samples, n_sources]
    """
    n_samples, n_sources, _ = s_true.shape
    sdr = np.zeros((n_samples, n_sources))
    for sample_idx in range(n_samples):
        for src_idx in range(n_sources):
            true = s_true[sample_idx, src_idx]
            hat = s_hat[sample_idx, src_idx]
            # 幅度对齐（解决盲分离的幅度不确定性）
            scale = np.sum(true * hat) / (np.sum(hat ** 2) + 1e-8)
            hat_scaled = hat * scale
            # 计算SDR
            signal_power = np.sum(true ** 2)
            noise_power = np.sum((true - hat_scaled) ** 2)
            if noise_power < 1e-10:
                sdr[sample_idx, src_idx] = 100.0
            else:
                sdr[sample_idx, src_idx] = 10 * np.log10(signal_power / noise_power)
    return sdr

def calculate_corr(s_true, s_hat):
    """
    计算皮尔逊相关系数
    :param s_true: 真实源信号，形状 [n_samples, n_sources, seq_len]
    :param s_hat: 估计源信号，形状同上
    :return: 每个样本每个源的相关系数，形状 [n_samples, n_sources]
    """
    n_samples, n_sources, _ = s_true.shape
    corr = np.zeros((n_samples, n_sources))
    for sample_idx in range(n_samples):
        for src_idx in range(n_sources):
            true = s_true[sample_idx, src_idx]
            hat = s_hat[sample_idx, src_idx]
            corr_coef = np.corrcoef(true, hat)[0, 1]
            corr[sample_idx, src_idx] = np.abs(corr_coef)
    return corr


# ====================== 论文标准：全局拒绝水平 GRL ======================
def calculate_grl(s_true, s_hat, eps=1e-12):
    """
    基于真实源与估计源计算样本级全局矩阵 G，再按论文公式计算平均 GRL。
    :param s_true: 真实源信号，形状 [n_samples, n_sources, seq_len]
    :param s_hat: 估计源信号，形状同上
    :return: 数据集平均 GRL，越小越好
    """
    n_samples, n_sources, _ = s_true.shape
    grl_values = []
    eye_eps = 1e-8
    for sample_idx in range(n_samples):
        true = s_true[sample_idx].reshape(n_sources, -1)
        hat = s_hat[sample_idx].reshape(n_sources, -1)
        r_ss = true @ true.T + eye_eps * np.eye(n_sources)
        g = hat @ true.T @ np.linalg.inv(r_ss)
        g_power = np.abs(g) ** 2
        row_sum = np.sum(g_power, axis=1)
        row_max = np.max(g_power, axis=1)
        row_term = np.sum(row_sum / (row_max + eps) - 1.0)
        col_sum = np.sum(g_power, axis=0)
        col_max = np.max(g_power, axis=0)
        col_term = np.sum(col_sum / (col_max + eps) - 1.0)
        grl_values.append(float(row_term + col_term))
    return float(np.mean(grl_values)) if grl_values else 0.0
