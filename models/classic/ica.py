import numpy as np
from scipy import linalg

def fastica(X, n_components=None, max_iter=200, tol=1e-6):
    """
    FastICA算法实现（基于Hyvarinen固定点算法）
    匹配毕设参考文献：A. Hyvarinen, 1999 Fast and Robust Fixed-Point Algorithms for ICA

    Parameters
    ----------
    X : np.ndarray
        混合信号矩阵，形状：[序列长度, 通道数]
    n_components : int, optional
        要分离的源信号数量，默认等于通道数
    max_iter : int, optional
        最大迭代次数，默认200
    tol : float, optional
        收敛阈值，默认1e-6

    Returns
    -------
    S : np.ndarray
        分离后的源信号，形状：[序列长度, 源数]
    W : np.ndarray
        分离矩阵
    """
    # 数据维度检查
    if X.ndim != 2:
        raise ValueError("输入X必须是二维矩阵，形状为[序列长度, 通道数]")
    
    n_samples, n_channels = X.shape
    if n_components is None:
        n_components = n_channels
    if n_components > n_channels:
        raise ValueError("源信号数量不能大于混合信号通道数")

    # 步骤1：数据中心化
    X = X - np.mean(X, axis=0)

    # 步骤2：数据白化
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    # 取最大的n_components个特征值
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:n_components]]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    # 白化变换
    whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
    X_white = np.dot(X, whitening_matrix)

    # 步骤3：固定点迭代
    W = np.random.randn(n_components, n_components)
    W = linalg.orth(W.T).T  # 正交化初始化

    for iter_idx in range(max_iter):
        W_old = W.copy()
        # 非线性变换（tanh函数，经典FastICA选择）
        wx = np.dot(W, X_white.T)
        g_wx = np.tanh(wx)
        g_prime_wx = 1 - g_wx ** 2
        # 固定点更新
        W_new = np.dot(g_wx, X_white) / n_samples - np.dot(np.diag(g_prime_wx.mean(axis=1)), W)
        # 正交化
        W = linalg.orth(W_new.T).T
        # 收敛判断
        if np.max(np.abs(np.abs(np.diag(np.dot(W, W_old.T))) - 1)) < tol:
            break

    # 步骤4：计算分离信号
    S = np.dot(W, X_white.T).T
    return S, W