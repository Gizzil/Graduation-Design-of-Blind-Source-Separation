import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.global_config import (
    TRAIN_CONFIG_PATH, MODEL_CONFIG_PATH, PROCESSED_DATA_DIR, 
    MODEL_SAVE_DIR, DEVICE, SEED
)
from utils.data_utils import BSSDataset
from utils.train_utils import EarlyStopping, save_model
from models.deep.cnn_sep import CNN_SEP
from models.deep.mlp_sep import MLP_SEP
from models.deep.tcn_sep import TCN_SEP

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    # 读取配置文件
    with open(TRAIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    print("="*40)
    print(f"模型训练启动，设备：{DEVICE}")
    print(f"模型类型：{train_config['model_type']}")
    print(f"训练轮数：{train_config['epochs']}")
    print(f"Batch Size：{train_config['batch_size']}")
    print(f"学习率：{train_config['lr']}")
    print("="*40)

    # 1. 加载数据集
    print("1. 加载训练数据集...")
    train_data = np.load(f"{PROCESSED_DATA_DIR}/train_data.npy")
    train_label = np.load(f"{PROCESSED_DATA_DIR}/train_label.npy")
    test_data = np.load(f"{PROCESSED_DATA_DIR}/test_data.npy")
    test_label = np.load(f"{PROCESSED_DATA_DIR}/test_label.npy")

    # 构建DataLoader
    train_dataset = BSSDataset(train_data, train_label)
    test_dataset = BSSDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # 2. 初始化模型
    print("2. 初始化分离模型...")
    n_mic = model_config['n_mic']
    n_source = model_config['n_source']
    seq_len = model_config['seq_len']

    if train_config['model_type'] == 'cnn':
        model = CNN_SEP(in_channels=n_mic, out_channels=n_source, seq_len=seq_len).to(DEVICE)
    elif train_config['model_type'] == 'mlp':
        model = MLP_SEP(in_dim=n_mic, out_dim=n_source, seq_len=seq_len).to(DEVICE)
    elif train_config['model_type'] == 'tcn':
        model = TCN_SEP(in_channels=n_mic, out_channels=n_source, seq_len=seq_len).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型类型：{train_config['model_type']}，可选cnn/mlp/tcn")
    
    # 3. 初始化优化器、损失函数、早停机制
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['l2_lambda']  # L2正则化
    )
    loss_fn = nn.MSELoss()  # MSE损失函数
    early_stopping = EarlyStopping(
        patience=train_config['patience'],
        min_delta=train_config['min_delta'],
        save_path=f"{MODEL_SAVE_DIR}/best_model.pth"
    )

    # 4. 训练循环
    print("3. 开始模型训练...")
    train_loss_list = []
    val_loss_list = []

    for epoch in range(train_config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']} 训练")
        for batch_data, batch_label in train_pbar:
            batch_data = batch_data.to(DEVICE)
            batch_label = batch_label.to(DEVICE)

            # 前向传播
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"训练损失": f"{loss.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']} 验证")
            for batch_data, batch_label in val_pbar:
                batch_data = batch_data.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                outputs = model(batch_data)
                loss = loss_fn(outputs, batch_label)
                val_loss += loss.item()
                val_pbar.set_postfix({"验证损失": f"{loss.item():.6f}"})
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch {epoch+1} 完成 | 平均训练损失：{avg_train_loss:.6f} | 平均验证损失：{avg_val_loss:.6f}")

        # 早停判断
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("早停触发，训练提前结束！")
            break

    # 保存最终模型与损失曲线
    save_model(model, f"{MODEL_SAVE_DIR}/final_model.pth")
    np.save(f"{MODEL_SAVE_DIR}/train_loss.npy", np.array(train_loss_list))
    np.save(f"{MODEL_SAVE_DIR}/val_loss.npy", np.array(val_loss_list))

    print("="*40)
    print("模型训练完成！最优模型已保存至 results/models/")
    print("="*40)