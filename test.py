import itertools

import numpy as np
import pandas as pd
import torch
import yaml

from configs.global_config import (
    DATA_CONFIG_PATH,
    DEVICE,
    FIGURE_SAVE_DIR,
    METRICS_SAVE_DIR,
    MIXED_DATA_DIR,
    MODEL_CONFIG_PATH,
    MODEL_SAVE_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TRAIN_CONFIG_PATH,
)
from models.classic.ica import fastica
from models.classic.joint_diag import sobi
from models.deep.cnn_sep import CNN_SEP
from models.deep.mlp_sep import MLP_SEP
from models.deep.tcn_sep import TCN_SEP
from utils.eval_utils import calculate_corr, calculate_grl, calculate_mse, calculate_sdr
from utils.vis_utils import (
    plot_loss_curve,
    plot_metrics_comparison,
    plot_robustness_curve,
    plot_signal_comparison,
    plot_time_freq,
)

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)


def standardize_mix(x, eps=1e-8):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def align_sources(true_sources, est_sources, eps=1e-8):
    n_source = true_sources.shape[0]
    best_perm = None
    best_score = -np.inf
    true_centered = true_sources - np.mean(true_sources, axis=1, keepdims=True)
    est_centered = est_sources - np.mean(est_sources, axis=1, keepdims=True)
    true_norm = np.sqrt(np.sum(true_centered**2, axis=1)) + eps
    est_norm = np.sqrt(np.sum(est_centered**2, axis=1)) + eps
    corr_mat = np.zeros((n_source, n_source))
    for i in range(n_source):
        for j in range(n_source):
            corr_mat[i, j] = np.abs(np.sum(true_centered[i] * est_centered[j]) / (true_norm[i] * est_norm[j]))
    for perm in itertools.permutations(range(n_source)):
        score = sum(corr_mat[i, perm[i]] for i in range(n_source))
        if score > best_score:
            best_score = score
            best_perm = perm
    aligned = est_sources[list(best_perm)]
    scales = np.sum(true_sources * aligned, axis=1) / (np.sum(aligned**2, axis=1) + eps)
    return aligned * scales[:, None]


def align_batch(true_batch, est_batch):
    aligned = np.zeros_like(est_batch)
    for i in range(est_batch.shape[0]):
        aligned[i] = align_sources(true_batch[i], est_batch[i])
    return aligned


def load_configs():
    with open(DATA_CONFIG_PATH, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    with open(TRAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    return data_config, model_config, train_config


def build_model(train_config, model_config):
    model_type = train_config["model_type"]
    if model_type == "cnn":
        model = CNN_SEP(
            in_channels=model_config["n_mic"],
            out_channels=model_config["n_source"],
            seq_len=model_config["seq_len"],
        )
    elif model_type == "mlp":
        model = MLP_SEP(
            in_dim=model_config["n_mic"],
            out_dim=model_config["n_source"],
            seq_len=model_config["seq_len"],
        )
    elif model_type == "tcn":
        model = TCN_SEP(
            in_channels=model_config["n_mic"],
            out_channels=model_config["n_source"],
            seq_len=model_config["seq_len"],
        )
    else:
        raise ValueError(f"不支持的模型类型：{model_type}，可选cnn/mlp/tcn")
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_SAVE_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    return model


def deep_inference(model, mixed_np, batch_size=512):
    preds = []
    with torch.no_grad():
        for start in range(0, mixed_np.shape[0], batch_size):
            end = min(start + batch_size, mixed_np.shape[0])
            batch = torch.from_numpy(mixed_np[start:end]).float().to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_classic_algorithms(test_data, test_label, n_source, max_samples=None):
    n_test = test_data.shape[0] if max_samples is None else min(test_data.shape[0], max_samples)
    ica_result = np.zeros_like(test_label)
    sobi_result = np.zeros_like(test_label)
    sobi_lags = min(200, test_data.shape[-1] // 4)

    for i in range(n_test):
        x = standardize_mix(test_data[i])
        ica_sep, _ = fastica(x.T, n_components=n_source)
        ica_result[i] = align_sources(test_label[i], ica_sep.T)
        sobi_sep, _ = sobi(x.T, n_components=n_source, n_lags=sobi_lags)
        sobi_result[i] = align_sources(test_label[i], sobi_sep.T)

    if n_test < test_data.shape[0]:
        ica_result[n_test:] = np.nan
        sobi_result[n_test:] = np.nan
    return ica_result, sobi_result


def collect_metrics(true_signals, separated_dict, n_source):
    metrics_result = []
    for alg, sep_signals in separated_dict.items():
        valid_mask = ~np.isnan(sep_signals).any(axis=(1, 2))
        sep_valid = sep_signals[valid_mask]
        true_valid = true_signals[valid_mask]
        mse = calculate_mse(true_valid, sep_valid)
        sdr = calculate_sdr(true_valid, sep_valid)
        corr = calculate_corr(true_valid, sep_valid)
        grl = calculate_grl(true_valid, sep_valid)

        for src_idx in range(n_source):
            metrics_result.append(
                {
                    "算法": alg,
                    "源信号序号": src_idx + 1,
                    "MSE": np.mean(mse[:, src_idx]),
                    "SDR(dB)": np.mean(sdr[:, src_idx]),
                    "相关系数": np.mean(corr[:, src_idx]),
                    "全局拒绝水平(GRL)": grl,
                }
            )
    return pd.DataFrame(metrics_result)


def evaluate_robustness(model, snr_list, test_source, n_source, run_classic=True, max_samples=None):
    robustness_result = []
    n_test = test_source.shape[0]

    for snr in snr_list:
        mixed_noisy = np.load(f"{MIXED_DATA_DIR}/mixed_signals_snr{snr}.npy")
        split_idx_full = mixed_noisy.shape[0] - n_test
        test_mixed = mixed_noisy[split_idx_full:]
        loop_len = test_mixed.shape[0] if max_samples is None else min(test_mixed.shape[0], max_samples)
        test_mixed = test_mixed[:loop_len]
        cur_source = test_source[:loop_len]

        deep_sep = deep_inference(model, test_mixed)
        deep_sep = align_batch(cur_source, deep_sep)
        deep_sdr = float(np.mean(calculate_sdr(cur_source, deep_sep)))

        row = {
            "SNR(dB)": snr,
            "深度学习模型": deep_sdr,
        }
        if run_classic:
            sobi_lags = min(200, test_mixed.shape[-1] // 4)
            ica_sdr_list = []
            sobi_sdr_list = []
            for i in range(loop_len):
                x = standardize_mix(test_mixed[i])
                ica_sep, _ = fastica(x.T, n_components=n_source)
                ica_sep = align_sources(cur_source[i], ica_sep.T)
                ica_sdr_list.append(np.mean(calculate_sdr(cur_source[i : i + 1], ica_sep[np.newaxis, :, :])))
                sobi_sep, _ = sobi(x.T, n_components=n_source, n_lags=sobi_lags)
                sobi_sep = align_sources(cur_source[i], sobi_sep.T)
                sobi_sdr_list.append(np.mean(calculate_sdr(cur_source[i : i + 1], sobi_sep[np.newaxis, :, :])))
            row["FastICA"] = float(np.mean(ica_sdr_list))
            row["SOBI"] = float(np.mean(sobi_sdr_list))
        robustness_result.append(row)
    return pd.DataFrame(robustness_result)


if __name__ == "__main__":
    # 可按需修改以下开关，减少冗余耗时流程
    RUN_CLASSIC = True
    RUN_ROBUSTNESS = True
    RUN_VISUALIZATION = True
    MAX_CLASSIC_SAMPLES = 10  # 例如改成 300 可加快经典算法评估
    MAX_ROBUSTNESS_SAMPLES = 10  # 例如改成 300 可加快鲁棒性测试

    data_config, model_config, train_config = load_configs()

    print("=" * 40)
    print("盲源分离算法对比测试启动")
    print("对比算法：FastICA | SOBI | 深度学习模型")
    print("=" * 40)
    print(f"运行设备：{DEVICE}")
    print(
        f"开关配置：classic={RUN_CLASSIC}, robustness={RUN_ROBUSTNESS}, "
        f"visualization={RUN_VISUALIZATION}"
    )

    print("1. 加载测试数据与训练好的模型...")
    test_data = np.load(f"{PROCESSED_DATA_DIR}/test_data.npy")
    test_label = np.load(f"{PROCESSED_DATA_DIR}/test_label.npy")
    n_source = data_config["n_source"]
    model = build_model(train_config, model_config)

    print("2. 执行各算法分离测试...")
    deep_separated = deep_inference(model, test_data)
    deep_aligned = align_batch(test_label, deep_separated)

    all_separated = {
        "source": test_label,
        "mixed": test_data,
        "deep": deep_aligned,
    }
    if RUN_CLASSIC:
        print("  经典算法推理中...")
        ica_sep, sobi_sep = run_classic_algorithms(
            test_data=test_data,
            test_label=test_label,
            n_source=n_source,
            max_samples=MAX_CLASSIC_SAMPLES,
        )
        all_separated["ica"] = ica_sep
        all_separated["sobi"] = sobi_sep
        print("  经典算法推理完成！")

    print("3. 计算分离效果量化指标...")
    metrics_targets = {"deep": all_separated["deep"]}
    if RUN_CLASSIC:
        metrics_targets["ica"] = all_separated["ica"]
        metrics_targets["sobi"] = all_separated["sobi"]
    metrics_df = collect_metrics(test_label, metrics_targets, n_source)
    metrics_df.to_csv(f"{METRICS_SAVE_DIR}/separation_metrics.csv", index=False, encoding="utf-8-sig")
    print("  量化指标已保存至 results/metrics/")
    print("\n分离效果平均指标：")
    print(metrics_df.groupby("算法").mean(numeric_only=True))

    robustness_df = None
    if RUN_ROBUSTNESS:
        print("\n4. 不同噪声强度鲁棒性测试...")
        robustness_df = evaluate_robustness(
            model=model,
            snr_list=data_config["snr_list"],
            test_source=test_label,
            n_source=n_source,
            run_classic=RUN_CLASSIC,
            max_samples=MAX_ROBUSTNESS_SAMPLES,
        )
        robustness_df.to_csv(
            f"{METRICS_SAVE_DIR}/robustness_metrics.csv", index=False, encoding="utf-8-sig"
        )
        print("  鲁棒性测试结果已保存！")

    if RUN_VISUALIZATION:
        print("\n5. 生成可视化结果图...")
        train_loss = np.load(f"{MODEL_SAVE_DIR}/train_loss.npy")
        val_loss = np.load(f"{MODEL_SAVE_DIR}/val_loss.npy")
        plot_loss_curve(train_loss, val_loss, save_path=f"{FIGURE_SAVE_DIR}/loss_curve.png")

        sample_idx = 0
        source_sample = all_separated["source"][sample_idx]
        mixed_sample = all_separated["mixed"][sample_idx]
        separated_signals = {"深度学习模型": all_separated["deep"][sample_idx]}
        if RUN_CLASSIC:
            separated_signals = {
                "FastICA": all_separated["ica"][sample_idx],
                "SOBI": all_separated["sobi"][sample_idx],
                "深度学习模型": all_separated["deep"][sample_idx],
            }
        plot_signal_comparison(
            source_signals=source_sample,
            mixed_signals=mixed_sample,
            separated_signals=separated_signals,
            fs=data_config["fs"],
            save_path=f"{FIGURE_SAVE_DIR}/signal_comparison.png",
        )

        src_labels = [f"源信号{i+1}" for i in range(source_sample.shape[0])]
        mix_labels = [f"混合信号{i+1}" for i in range(mixed_sample.shape[0])]
        fs = data_config["fs"]
        plot_time_freq(
            signals=source_sample,
            fs=fs,
            channel_labels=src_labels,
            suptitle="源信号时域与频域",
            save_path=f"{FIGURE_SAVE_DIR}/source_time_freq.png",
        )
        plot_time_freq(
            signals=mixed_sample,
            fs=fs,
            channel_labels=mix_labels,
            suptitle="混合信号时域与频域",
            save_path=f"{FIGURE_SAVE_DIR}/mixed_time_freq.png",
        )
        if RUN_CLASSIC:
            demix_labels = [f"解混信号{i+1}" for i in range(all_separated["ica"][sample_idx].shape[0])]
            plot_time_freq(
                signals=all_separated["ica"][sample_idx],
                fs=fs,
                channel_labels=demix_labels,
                suptitle="FastICA 解混信号时域与频域",
                save_path=f"{FIGURE_SAVE_DIR}/ica_time_freq.png",
            )
            plot_time_freq(
                signals=all_separated["sobi"][sample_idx],
                fs=fs,
                channel_labels=demix_labels,
                suptitle="SOBI 解混信号时域与频域",
                save_path=f"{FIGURE_SAVE_DIR}/sobi_time_freq.png",
            )
            plot_time_freq(
                signals=all_separated["deep"][sample_idx],
                fs=fs,
                channel_labels=demix_labels,
                suptitle="深度学习模型解混信号时域与频域",
                save_path=f"{FIGURE_SAVE_DIR}/deep_time_freq.png",
            )
        plot_metrics_comparison(metrics_df, save_path=f"{FIGURE_SAVE_DIR}/metrics_comparison.png")
        if robustness_df is not None and not robustness_df.empty:
            plot_robustness_curve(robustness_df, save_path=f"{FIGURE_SAVE_DIR}/robustness_curve.png")
        print("  可视化图已保存至 results/figures/")

    print("\n" + "=" * 40)
    print("所有测试与评估完成！结果已全部保存")
    print("=" * 40)
