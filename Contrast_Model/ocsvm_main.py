import numpy as np
import os
from data_preparation import UnifiedTimeSeriesDatasetGenerator, DATASET_CONFIG
from train_eval import set_graded_seed, evaluate_model
from ocsvm_model import OCSVMAnomalyDetector

# ========== 新增：导入标准VUS计算函数 ==========
from vus_calculator import get_standard_vus


def main():
    # 全局配置（与GRU保持一致，便于对比）
    global_config = {
        "K": 10,
        "n_runs": 3,  # 多次运行取均值
        "device": "cpu",  # OCSVM仅支持CPU
        "datasets": [],
        "dataset_sigma": {
            "smd": 3.0,
            "smap_msl": 2.0,
            "mackey_glass": 2.5
        },
        # OCSVM专属配置
        "ocsvm_params": {
            "nu": 0.05,  # 对应数据污染率0.05
            "kernel": "rbf",
            "gamma": "scale"
        }
    }

    print(f"🚀 One-Class SVM异常检测模型训练器 | 使用设备：{global_config['device']}")
    print(f"⚙️  OCSVM配置：nu={global_config['ocsvm_params']['nu']}, kernel={global_config['ocsvm_params']['kernel']}")

    # 交互式选择数据集
    available_datasets = list(DATASET_CONFIG.keys())
    print(f"\n📋 可选数据集：{', '.join(available_datasets)}")

    while True:
        user_input = input("\n请输入要训练的数据集名称（多个用逗号分隔，如smd,smap_msl）：").strip()
        selected_datasets = [ds.strip().lower() for ds in user_input.split(",") if ds.strip()]

        invalid_datasets = [ds for ds in selected_datasets if ds not in available_datasets]
        if not selected_datasets:
            print("❌ 输入不能为空，请重新输入！")
        elif invalid_datasets:
            print(f"❌ 无效数据集名称：{', '.join(invalid_datasets)}，可选数据集：{', '.join(available_datasets)}")
        else:
            global_config["datasets"] = selected_datasets
            print(f"✅ 已选择训练数据集：{', '.join(selected_datasets)}")
            break

    # 存储结果
    all_dataset_results = {}

    # 遍历数据集
    for dataset_name in global_config["datasets"]:
        print(f"\n{'=' * 100}")
        print(f"📌 开始测试 {dataset_name.upper()} 数据集")
        print(f"{'=' * 100}")

        # SMAP_MSL单文件处理
        if dataset_name == "smap_msl":
            generator = UnifiedTimeSeriesDatasetGenerator(dataset_name=dataset_name)
            test_files = generator.get_smap_msl_test_files()
            test_files = [f for f in test_files if os.path.basename(f).startswith("E-1")]
            if len(test_files) == 0:
                print(f"❌ SMAP_MSL 无测试文件")
                continue

            smap_msl_file_results = {}
            for file_path in test_files:
                print(f"\n{'=' * 80}")
                print(f"📄 开始测试 SMAP_MSL 文件：{os.path.basename(file_path)}")
                print(f"{'=' * 80}")

                try:
                    file_generator = UnifiedTimeSeriesDatasetGenerator(
                        dataset_name=dataset_name,
                        window_length=DATASET_CONFIG[dataset_name]["window_length"],
                        anomaly_contamination=global_config["ocsvm_params"]["nu"],
                        file_path=file_path
                    )
                    train_windows, test_windows, train_labels, test_labels, n_vars = file_generator.prepare_dataset()

                    # 数据预处理：清理NaN/Inf
                    train_windows = np.nan_to_num(train_windows, nan=0.0, posinf=1e6, neginf=-1e6)
                    test_windows = np.nan_to_num(test_windows, nan=0.0, posinf=1e6, neginf=-1e6)

                    if len(train_windows) == 0 or len(test_windows) == 0:
                        print(f"⚠️ 跳过文件 {os.path.basename(file_path)}：窗口数为0")
                        continue
                    print(
                        f"✅ 加载 {os.path.basename(file_path)} 成功 | 训练窗口数：{len(train_windows)} | 测试窗口数：{len(test_windows)} | 特征数：{n_vars}")
                except Exception as e:
                    print(f"❌ 加载 {os.path.basename(file_path)} 失败：{str(e)}")
                    continue

                # 多次运行
                file_metrics = []
                # ========== 新增：初始化VUS存储列表 ==========
                file_vus_roc = []
                file_vus_pr = []

                for run_idx in range(global_config["n_runs"]):
                    print(f"\n--- {os.path.basename(file_path)} 第 {run_idx + 1}/{global_config['n_runs']} 次运行 ---")

                    # 固定种子
                    set_graded_seed(base_seed=42 + run_idx)

                    # 初始化OCSVM检测器
                    detector = OCSVMAnomalyDetector(
                        nu=global_config["ocsvm_params"]["nu"],
                        kernel=global_config["ocsvm_params"]["kernel"],
                        gamma=global_config["ocsvm_params"]["gamma"],
                        random_state=42 + run_idx
                    )

                    # 训练OCSVM
                    print(f"📈 开始训练OCSVM | nu={detector.nu}, kernel={detector.kernel}")
                    detector.train(train_windows)

                    # 预测
                    print(f"🔍 开始预测异常分数")
                    anomaly_scores = detector.predict(test_windows)
                    print(f"✅ 预测完成 | 异常分数长度：{len(anomaly_scores)}")

                    # ========== 新增：计算标准VUS指标 ==========
                    vus_roc, vus_pr = get_standard_vus(anomaly_scores, test_labels)
                    file_vus_roc.append(vus_roc)
                    file_vus_pr.append(vus_pr)
                    print(f"📊 第{run_idx + 1}次运行 VUS-ROC: {vus_roc:.4f}, VUS-PR: {vus_pr:.4f}")

                    # 评估（使用统一的evaluate_model函数，与GRU对齐）
                    eval_results = evaluate_model(
                        anomaly_scores,
                        test_labels,
                        dataset_name=f"{dataset_name}_{os.path.basename(file_path).split('.')[0]}",
                        plot=(run_idx == global_config["n_runs"] - 1),
                        target_precision=0.8
                    )
                    file_metrics.append(eval_results)

                # 计算均值±标准差
                def calc_mean_std(metric_name):
                    values = [m[metric_name] for m in file_metrics]
                    return np.mean(values), np.std(values)

                # 指标汇总
                precision_mean, precision_std = calc_mean_std("best_precision")
                recall_mean, recall_std = calc_mean_std("best_recall")
                f1_mean, f1_std = calc_mean_std("best_f1")
                auroc_mean, auroc_std = calc_mean_std("auc_roc")
                aupr_mean, aupr_std = calc_mean_std("auc_pr")
                variance_gap_mean, variance_gap_std = calc_mean_std("variance_gap") if "variance_gap" in file_metrics[
                    0] else (0.0, 0.0)
                dynamic_f1_mean, dynamic_f1_std = calc_mean_std("dynamic_f1")

                # ========== 新增：计算VUS均值±标准差 ==========
                vus_roc_mean, vus_roc_std = np.mean(file_vus_roc), np.std(file_vus_roc)
                vus_pr_mean, vus_pr_std = np.mean(file_vus_pr), np.std(file_vus_pr)

                # 存储结果
                smap_msl_file_results[os.path.basename(file_path)] = {
                    "Precision": (precision_mean, precision_std),
                    "Recall": (recall_mean, recall_std),
                    "F1 Score": (f1_mean, f1_std),
                    "Dynamic F1": (dynamic_f1_mean, dynamic_f1_std),
                    "AUROC": (auroc_mean, auroc_std),
                    "AUPR": (aupr_mean, aupr_std),
                    "Variance Gap": (variance_gap_mean, variance_gap_std),
                    # ========== 新增：存储VUS指标 ==========
                    "VUS-ROC": (round(vus_roc_mean, 4), round(vus_roc_std, 4)),
                    "VUS-PR": (round(vus_pr_mean, 4), round(vus_pr_std, 4))
                }

                # 打印结果
                print(f"\n🏆 {os.path.basename(file_path)} 最终结果（{global_config['n_runs']}次运行均值±标准差）")
                print(f"静态阈值 F1: {f1_mean:.4f} ± {f1_std:.4f}")
                print(f"动态阈值 F1: {dynamic_f1_mean:.4f} ± {dynamic_f1_std:.4f}")
                print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
                print(f"Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
                print(f"AUROC:     {auroc_mean:.4f} ± {auroc_std:.4f}")
                print(f"AUPR:      {aupr_mean:.4f} ± {aupr_std:.4f}")
                # ========== 新增：打印VUS结果 ==========
                print(f"VUS-ROC:   {vus_roc_mean:.4f} ± {vus_roc_std:.4f}")
                print(f"VUS-PR:    {vus_pr_mean:.4f} ± {vus_pr_std:.4f}")
                if variance_gap_mean != 0.0:
                    print(f"Variance Gap: {variance_gap_mean:.4f} ± {variance_gap_std:.4f}")

            all_dataset_results[dataset_name] = smap_msl_file_results
            continue

        # 其他数据集处理（SMD/Mackey-Glass）
        try:
            generator = UnifiedTimeSeriesDatasetGenerator(
                dataset_name=dataset_name,
                window_length=DATASET_CONFIG[dataset_name]["window_length"],
                anomaly_contamination=global_config["ocsvm_params"]["nu"]
            )
            if dataset_name == "mackey_glass":
                print(f"🔄 开始生成 Mackey-Glass 数据集（代码生成）")
            else:
                print(f"🔄 开始加载 {dataset_name} 数据集（本地文件）")

            train_windows, test_windows, train_labels, test_labels, n_vars = generator.prepare_dataset()

            # 数据预处理：清理NaN/Inf
            train_windows = np.nan_to_num(train_windows, nan=0.0, posinf=1e6, neginf=-1e6)
            test_windows = np.nan_to_num(test_windows, nan=0.0, posinf=1e6, neginf=-1e6)

            print(
                f"✅ {dataset_name} 数据加载/生成成功 | 训练窗口数：{len(train_windows)} | 测试窗口数：{len(test_windows)} | 特征数：{n_vars}")
        except Exception as e:
            print(f"❌ 处理 {dataset_name} 失败：{str(e)}")
            continue

        # 多次运行
        dataset_metrics = []
        # ========== 新增：初始化VUS存储列表 ==========
        dataset_vus_roc = []
        dataset_vus_pr = []

        for run_idx in range(global_config["n_runs"]):
            print(f"\n--- {dataset_name} 第 {run_idx + 1}/{global_config['n_runs']} 次运行 ---")

            # 固定种子
            set_graded_seed(base_seed=42 + run_idx)

            # 初始化OCSVM
            detector = OCSVMAnomalyDetector(
                nu=global_config["ocsvm_params"]["nu"],
                kernel=global_config["ocsvm_params"]["kernel"],
                gamma=global_config["ocsvm_params"]["gamma"],
                random_state=42 + run_idx
            )

            # 训练
            print(f"📈 开始训练OCSVM | nu={detector.nu}, kernel={detector.kernel}")
            detector.train(train_windows)

            # 预测
            print(f"🔍 开始预测异常分数")
            anomaly_scores = detector.predict(test_windows)
            print(f"✅ 预测完成 | 异常分数长度：{len(anomaly_scores)}")

            # ========== 新增：计算标准VUS指标 ==========
            vus_roc, vus_pr = get_standard_vus(anomaly_scores, test_labels)
            dataset_vus_roc.append(vus_roc)
            dataset_vus_pr.append(vus_pr)
            print(f"📊 第{run_idx + 1}次运行 VUS-ROC: {vus_roc:.4f}, VUS-PR: {vus_pr:.4f}")

            # 评估
            eval_results = evaluate_model(
                anomaly_scores,
                test_labels,
                dataset_name=dataset_name,
                plot=(run_idx == global_config["n_runs"] - 1),
                target_precision=0.8
            )
            dataset_metrics.append(eval_results)

        # 计算均值±标准差
        def calc_mean_std(metric_name):
            values = [m[metric_name] for m in dataset_metrics]
            return np.mean(values), np.std(values)

        # 指标汇总
        precision_mean, precision_std = calc_mean_std("best_precision")
        recall_mean, recall_std = calc_mean_std("best_recall")
        f1_mean, f1_std = calc_mean_std("best_f1")
        auroc_mean, auroc_std = calc_mean_std("auc_roc")
        aupr_mean, aupr_std = calc_mean_std("auc_pr")
        variance_gap_mean, variance_gap_std = calc_mean_std("variance_gap") if "variance_gap" in dataset_metrics[
            0] else (0.0, 0.0)
        dynamic_f1_mean, dynamic_f1_std = calc_mean_std("dynamic_f1")

        # ========== 新增：计算VUS均值±标准差 ==========
        vus_roc_mean, vus_roc_std = np.mean(dataset_vus_roc), np.std(dataset_vus_roc)
        vus_pr_mean, vus_pr_std = np.mean(dataset_vus_pr), np.std(dataset_vus_pr)

        # 存储结果
        all_dataset_results[dataset_name] = {
            "Precision": (precision_mean, precision_std),
            "Recall": (recall_mean, recall_std),
            "F1 Score": (f1_mean, f1_std),
            "Dynamic F1": (dynamic_f1_mean, dynamic_f1_std),
            "AUROC": (auroc_mean, auroc_std),
            "AUPR": (aupr_mean, aupr_std),
            "Variance Gap": (variance_gap_mean, variance_gap_std),
            # ========== 新增：存储VUS指标 ==========
            "VUS-ROC": (round(vus_roc_mean, 4), round(vus_roc_std, 4)),
            "VUS-PR": (round(vus_pr_mean, 4), round(vus_pr_std, 4))
        }

        # 打印结果
        print(f"\n🏆 {dataset_name.upper()} 最终结果（{global_config['n_runs']}次运行均值±标准差）")
        print(f"静态阈值 F1: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"动态阈值 F1: {dynamic_f1_mean:.4f} ± {dynamic_f1_std:.4f}")
        print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"AUROC:     {auroc_mean:.4f} ± {auroc_std:.4f}")
        print(f"AUPR:      {aupr_mean:.4f} ± {aupr_std:.4f}")
        # ========== 新增：打印VUS结果 ==========
        print(f"VUS-ROC:   {vus_roc_mean:.4f} ± {vus_roc_std:.4f}")
        print(f"VUS-PR:    {vus_pr_mean:.4f} ± {vus_pr_std:.4f}")
        if variance_gap_mean != 0.0:
            print(f"Variance Gap: {variance_gap_mean:.4f} ± {variance_gap_std:.4f}")

    # 汇总输出
    print(f"\n{'=' * 100}")
    print(f"📊 目标数据集最终汇总结果（用户选择顺序）")
    print(f"{'=' * 100}")
    for ds_name in global_config["datasets"]:
        if ds_name not in all_dataset_results:
            print(f"\n{ds_name.upper()}: ❌ 处理失败，无结果")
            continue

        print(f"\n{ds_name.upper()}:")
        if ds_name == "smap_msl":
            for file_name, results in all_dataset_results[ds_name].items():
                print(f"  📄 {file_name}:")
                print(f"    静态阈值 F1: {results['F1 Score'][0]:.4f} ± {results['F1 Score'][1]:.4f}")
                print(f"    动态阈值 F1: {results['Dynamic F1'][0]:.4f} ± {results['Dynamic F1'][1]:.4f}")
                for metric, (mean_val, std_val) in results.items():
                    if metric not in ["F1 Score", "Dynamic F1"]:
                        print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            results = all_dataset_results[ds_name]
            print(f"  静态阈值 F1: {results['F1 Score'][0]:.4f} ± {results['F1 Score'][1]:.4f}")
            print(f"  动态阈值 F1: {results['Dynamic F1'][0]:.4f} ± {results['Dynamic F1'][1]:.4f}")
            for metric, (mean_val, std_val) in results.items():
                if metric not in ["F1 Score", "Dynamic F1"]:
                    print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()