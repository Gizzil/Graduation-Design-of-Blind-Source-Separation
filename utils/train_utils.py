import torch
import numpy as np

class EarlyStopping:
    """早停机制类，防止过拟合"""
    def __init__(self, patience=10, min_delta=1e-5, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 验证损失下降，更新最优模型
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            # 验证损失未下降，计数+1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_model(model, save_path):
    """保存模型权重"""
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path, device):
    """加载模型权重"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model