import numpy as np
import librosa
import os
import glob
import random
from torch.utils.data import Dataset
import wfdb
from scipy import signal

# ===================== 定义原始信号路径 =====================
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_SPEECH_DIR = os.path.join(ROOT_DIR, "data", "raw", "speech")
RAW_ECG_DIR = os.path.join(ROOT_DIR, "data", "raw", "ecg")

_speech_files_cache = None
_ecg_records_cache = None
_ecg_signals_cache = {}


def _get_speech_file_list():
    global _speech_files_cache
    if _speech_files_cache is None:
        pattern1 = os.path.join(RAW_SPEECH_DIR, "**", "*.wav")
        pattern2 = os.path.join(RAW_SPEECH_DIR, "**", "*.WAV")
        _speech_files_cache = glob.glob(pattern1, recursive=True) + glob.glob(pattern2, recursive=True)
        if len(_speech_files_cache) == 0:
            raise RuntimeError(f"未在 {RAW_SPEECH_DIR} 找到任何语音文件")
    return _speech_files_cache


def _get_ecg_record_list():
    global _ecg_records_cache
    if _ecg_records_cache is None:
        pattern = os.path.join(RAW_ECG_DIR, "**", "*.hea")
        all_files = glob.glob(pattern, recursive=True)
        rec_paths = sorted({os.path.splitext(f)[0] for f in all_files})
        if len(rec_paths) == 0:
            raise RuntimeError(f"未在 {RAW_ECG_DIR} 找到任何ECG记录")
        _ecg_records_cache = rec_paths
    return _ecg_records_cache


def load_random_speech_segment(fs, seq_len):
    file_list = _get_speech_file_list()
    wav_path = random.choice(file_list)
    audio, sr = librosa.load(wav_path, sr=fs)
    if len(audio) < seq_len:
        pad_len = seq_len - len(audio)
        audio = np.pad(audio, (0, pad_len))
    else:
        start = np.random.randint(0, len(audio) - seq_len + 1)
        audio = audio[start : start + seq_len]
    max_abs = np.max(np.abs(audio)) + 1e-8
    audio = audio / max_abs
    return audio


def load_random_ecg_segment(fs, seq_len):
    rec_paths = _get_ecg_record_list()
    rec_path = random.choice(rec_paths)
    if rec_path in _ecg_signals_cache:
        ecg, fs_orig = _ecg_signals_cache[rec_path]
    else:
        record = wfdb.rdrecord(rec_path)
        ecg = record.p_signal[:, 0]
        fs_orig = record.fs
        _ecg_signals_cache[rec_path] = (ecg, fs_orig)
    duration = seq_len / float(fs)
    window_len_orig = int(duration * fs_orig)
    if window_len_orig <= 0:
        window_len_orig = 1
    if len(ecg) <= window_len_orig:
        start = 0
    else:
        start = np.random.randint(0, len(ecg) - window_len_orig + 1)
    ecg_window = ecg[start : start + window_len_orig]
    ecg_resampled = signal.resample(ecg_window, seq_len)
    max_abs = np.max(np.abs(ecg_resampled)) + 1e-8
    ecg_resampled = ecg_resampled / max_abs
    return ecg_resampled

# ===================== 源信号生成函数 =====================
def generate_sine_signal(freq, fs, seq_len):
    """生成正弦信号"""
    t = np.linspace(0, seq_len/fs, seq_len)
    signal = np.sin(2 * np.pi * freq * t)
    return signal

def generate_bpsk_signal(fs, seq_len, symbol_rate=1000):
    """生成BPSK调制信号"""
    n_symbols = int(seq_len / fs * symbol_rate)
    symbols = np.random.choice([-1, 1], size=n_symbols)
    samples_per_symbol = int(fs / symbol_rate)
    signal = np.repeat(symbols, samples_per_symbol)
    if len(signal) < seq_len:
        signal = np.pad(signal, (0, seq_len - len(signal)))
    else:
        signal = signal[:seq_len]
    return signal

def generate_qpsk_signal(fs, seq_len, symbol_rate=1000):
    n_symbols = int(seq_len / fs * symbol_rate)
    if n_symbols <= 0:
        n_symbols = 1
    symbol_indices = np.random.randint(0, 4, size=n_symbols)
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128) / np.sqrt(2.0)
    symbols = constellation[symbol_indices]
    samples_per_symbol = int(fs / symbol_rate)
    if samples_per_symbol <= 0:
        samples_per_symbol = 1
    baseband = np.repeat(symbols, samples_per_symbol)
    if len(baseband) < seq_len:
        baseband = np.pad(baseband, (0, seq_len - len(baseband)))
    else:
        baseband = baseband[:seq_len]
    signal = np.real(baseband)
    return signal

def generate_speech_signal(fs, seq_len):
    """生成模拟语音信号（无需外部文件）"""
    signal = librosa.chirp(fmin=100, fmax=3000, sr=fs, duration=seq_len/fs)
    signal += np.random.randn(len(signal)) * 0.05
    return signal

def generate_source_signals(n_source, seq_len, n_samples, signal_types, fs):
    """
    批量生成多类型源信号
    :param n_source: 源信号数量
    :param seq_len: 单条信号长度
    :param n_samples: 样本数量
    :param signal_types: 每个源的信号类型
    :param fs: 采样率
    :return: 源信号矩阵，形状 [n_samples, n_source, seq_len]
    """
    source_signals = np.zeros((n_samples, n_source, seq_len))
    for sample_idx in range(n_samples):
        for src_idx in range(n_source):
            sig_type = signal_types[src_idx]
            if sig_type == 'sine':
                freq = np.random.uniform(100, 1000)
                sig = generate_sine_signal(freq, fs, seq_len)
            elif sig_type == 'bpsk':
                sig = generate_bpsk_signal(fs, seq_len)
            elif sig_type == 'qpsk':
                sig = generate_qpsk_signal(fs, seq_len)
            elif sig_type == "speech":
                sig = generate_speech_signal(fs, seq_len)
            elif sig_type == "speech_real":
                sig = load_random_speech_segment(fs, seq_len)
            elif sig_type == "ecg_real":
                sig = load_random_ecg_segment(fs, seq_len)
            else:
                raise ValueError(f"不支持的信号类型：{sig_type}")
            source_signals[sample_idx, src_idx] = sig
    return source_signals

# ===================== 混合与噪声函数 =====================
def generate_mixing_matrix(n_source, n_mic):
    """生成随机非奇异混合矩阵"""
    while True:
        A = np.random.randn(n_mic, n_source)
        if np.linalg.matrix_rank(A) == min(n_mic, n_source):
            break
    # 归一化
    A = A / np.max(np.abs(A))
    return A

def add_noise(signals, snr):
    """
    按指定SNR添加高斯白噪声
    :param signals: 原始信号，形状 [n_samples, n_channels, seq_len]
    :param snr: 信噪比（dB）
    :return: 加噪后的信号
    """
    snr_linear = 10 ** (snr / 10)
    signal_power = np.mean(signals ** 2, axis=(-1, -2), keepdims=True)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*signals.shape) * np.sqrt(noise_power)
    return signals + noise

# ===================== 数据预处理 =====================
def preprocess_data(mixed_signals, source_signals, train_ratio=0.8):
    """
    数据归一化与训练/测试集划分
    :param mixed_signals: 混合信号，形状 [n_samples, n_mic, seq_len]
    :param source_signals: 源信号，形状 [n_samples, n_source, seq_len]
    :param train_ratio: 训练集比例
    :return: train_data, test_data, train_label, test_label
    """
    # 按样本维度归一化
    data_mean = np.mean(mixed_signals, axis=(-1, -2), keepdims=True)
    data_std = np.std(mixed_signals, axis=(-1, -2), keepdims=True)
    mixed_signals = (mixed_signals - data_mean) / (data_std + 1e-8)

    label_mean = np.mean(source_signals, axis=(-1, -2), keepdims=True)
    label_std = np.std(source_signals, axis=(-1, -2), keepdims=True)
    source_signals = (source_signals - label_mean) / (label_std + 1e-8)

    # 划分训练/测试集
    n_samples = mixed_signals.shape[0]
    split_idx = int(n_samples * train_ratio)
    train_data = mixed_signals[:split_idx]
    test_data = mixed_signals[split_idx:]
    train_label = source_signals[:split_idx]
    test_label = source_signals[split_idx:]

    return train_data, test_data, train_label, test_label

def save_dataset(train_data, test_data, train_label, test_label, save_dir):
    """保存预处理后的数据集"""
    np.save(f"{save_dir}/train_data.npy", train_data)
    np.save(f"{save_dir}/test_data.npy", test_data)
    np.save(f"{save_dir}/train_label.npy", train_label)
    np.save(f"{save_dir}/test_label.npy", test_label)

# ===================== PyTorch Dataset类 =====================
class BSSDataset(Dataset):
    """盲源分离数据集类"""
    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
