import os
import torch

# ===================== 全局随机种子（保证实验可复现）=====================
SEED = 2026

# ===================== 计算设备配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 路径配置 =====================
# 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置文件路径
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "data", "data_config.yaml")
MODEL_CONFIG_PATH = os.path.join(ROOT_DIR, "models", "deep", "model_config.yaml")
TRAIN_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "train_config.yaml")

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MIXED_DATA_DIR = os.path.join(DATA_DIR, "mixed")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 结果目录
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_SAVE_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURE_SAVE_DIR = os.path.join(RESULTS_DIR, "figures")

# ===================== 目录创建函数 =====================
def make_dirs():
    """创建所有需要的目录"""
    dir_list = [
        RAW_DATA_DIR, MIXED_DATA_DIR, PROCESSED_DATA_DIR,
        MODEL_SAVE_DIR, METRICS_SAVE_DIR, FIGURE_SAVE_DIR
    ]
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)