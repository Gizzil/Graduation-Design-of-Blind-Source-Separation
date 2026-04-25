import yaml
import numpy as np
from configs.global_config import (
    DATA_CONFIG_PATH, RAW_DATA_DIR, MIXED_DATA_DIR, PROCESSED_DATA_DIR, SEED
)
from utils.data_utils import (
    generate_source_signals, generate_mixing_matrix, add_noise, preprocess_data, save_dataset
)

# 固定随机种子，保证实验可复现
np.random.seed(SEED)

if __name__ == "__main__":
    # 读取数据集配置
    with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*40)
    print("数据集生成配置：")
    print(f"源信号数量：{config['n_source']}")
    print(f"传感器数量：{config['n_mic']}")
    print(f"信号长度：{config['seq_len']}")
    print(f"样本数量：{config['n_samples']}")
    print(f"SNR范围：{config['snr_list']} dB")
    print("="*40)

    # 1. 生成原始源信号
    print("1. 生成原始源信号...")
    source_signals = generate_source_signals(
        n_source=config['n_source'],
        seq_len=config['seq_len'],
        n_samples=config['n_samples'],
        signal_types=config['signal_types'],
        fs=config['fs']
    )
    # 保存原始源信号
    np.save(f"{RAW_DATA_DIR}/source_signals.npy", source_signals)
    print(f"原始源信号已保存，形状：{source_signals.shape} [样本数, 源数, 序列长度]")

    # 2. 生成混合矩阵与混合信号
    print("2. 生成混合信号...")
    mixing_matrix = generate_mixing_matrix(
        n_source=config['n_source'],
        n_mic=config['n_mic']
    )
    np.save(f"{RAW_DATA_DIR}/mixing_matrix.npy", mixing_matrix)
    print(f"混合矩阵已保存，形状：{mixing_matrix.shape}")

    mixed_signals = np.einsum('ij, bjk -> bik', mixing_matrix, source_signals)

    print("3. 添加不同强度噪声...")
    mixed_signals_noisy = {}
    for snr in config['snr_list']:
        mixed_noisy = add_noise(mixed_signals, snr=snr)
        mixed_signals_noisy[snr] = mixed_noisy
        np.save(f"{MIXED_DATA_DIR}/mixed_signals_snr{snr}.npy", mixed_noisy)
        print(f"SNR={snr}dB 混合信号已保存，形状：{mixed_noisy.shape}")
        
    print("4. 数据预处理与数据集划分...")
    train_data, test_data, train_label, test_label = preprocess_data(
        mixed_signals=mixed_signals_noisy[config['base_snr']],
        source_signals=source_signals,
        train_ratio=config['train_test_split']
    )

    # 保存预处理后的数据集
    save_dataset(
        train_data=train_data,
        test_data=test_data,
        train_label=train_label,
        test_label=test_label,
        save_dir=PROCESSED_DATA_DIR
    )

    print("="*40)
    print("数据集生成全部完成！")
    print("="*40)
