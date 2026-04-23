import numpy as np
from scipy import linalg

def sobi(X, n_components=None, n_lags=100):
    """
    SOBI算法（基于二阶统计量的盲源分离）
    匹配毕设参考文献：S. Choi, 2002 Second Order Nonstationary Source Separation

    Parameters
    ----------
    X : np.ndarray
        混合信号矩阵，形状：[序列长度, 通道数]
    n_components : int, optional
        要分离的源信号数量，默认等于通道数
    n_lags : int, optional
        时延数量，默认100

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
    X = X.T  # 转换为[通道数, 序列长度]

    # 步骤2：白化
    cov = np.dot(X, X.T) / n_samples
    d, E = linalg.eigh(cov)
    idx = np.argsort(d)[::-1]
    d = d[idx[:n_components]]
    E = E[:, idx[:n_components]]
    whitening_matrix = np.dot(np.diag(1.0 / np.sqrt(d)), E.T)
    X_white = np.dot(whitening_matrix, X)

    # 步骤3：计算不同时延的协方差矩阵
    R_list = []
    for k in range(n_lags):
        if k == 0:
            R = np.dot(X_white[:, k:], X_white[:, k:].T) / (n_samples - k)
        else:
            R = np.dot(X_white[:, k:], X_white[:, :-k].T) / (n_samples - k)
        R_list.append(R)

    # 步骤4：联合对角化
    def joint_diag(R_list, max_iter=100, tol=1e-6):
        n = R_list[0].shape[0]
        V = np.eye(n)
        for iter_idx in range(max_iter):
            V_old = V.copy()
            for i in range(n):
                for j in range(i+1, n):
                    # 计算Givens旋转
                    g = np.zeros(2)
                    h = np.zeros(2)
                    for R in R_list:
                        R_ij = R[i, j]
                        R_ji = R[j, i]
                        R_ii = R[i, i]
                        R_jj = R[j, j]
                        g[0] += R_ij * (R_ii - R_jj)
                        g[1] += R_ji * (R_ii - R_jj)
                        h[0] += R_ii**2 + R_jj**2 - 2*R_ij*R_ji
                        h[1] += (R_ij**2 + R_ji**2) * 2
                    # 计算旋转角度
                    theta = 0.5 * np.arctan2(2 * np.dot(g, h), h[0]**2 - h[1]**2)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    # 应用旋转
                    G = np.array([[c, -s], [s, c]])
                    V[:, [i, j]] = np.dot(V[:, [i, j]], G)
                    for R in R_list:
                        R[[i, j], :] = np.dot(G.T, R[[i, j], :])
                        R[:, [i, j]] = np.dot(R[:, [i, j]], G)
            # 收敛判断
            if np.linalg.norm(V - V_old) < tol:
                break
        return V

    V = joint_diag(R_list)
    # 分离矩阵
    W = np.dot(V.T, whitening_matrix)
    # 分离信号
    S = np.dot(W, X).T
    return S, W