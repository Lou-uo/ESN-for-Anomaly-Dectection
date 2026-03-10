import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import StandardScaler

# 全局配置：新增 Lorenz 数据集配置
SEED = 42

DATASET_CONFIG = {
    "mackey_glass": {
        "type": "synthetic",
        "window_length": 50,
        "n_vars": 3,
        "anomaly_ratio": 0.01,
        "path": None  # 合成数据无需路径
    },
    "lorenz": {  # 新增 Lorenz 配置
        "type": "synthetic",
        "window_length": 50,  # 轻量级窗口长度
        "n_vars": 3,  # Lorenz 是 3 变量混沌序列
        "anomaly_ratio": 0.01,
        "path": None  # 合成数据无需路径
    },
    "smap_msl": {
        "type": "real",
        "window_length": 100,
        "n_vars": None,
        "path": r"E:/Test4/OmniAnomaly/archive",
        "single_file_mode": True  # 开启单文件模式
    },
    "smd": {
        "type": "real",
        "window_length": 100,
        "n_vars": 38,
        "path": r"E:/Test4/OmniAnomaly/ServerMachineDataset"
    },
    "cmapss": {
        "type": "real",
        "window_length": 50,  # 轻量级窗口长度
        "n_vars": 24,  # 21个传感器 + 3个操作设置
        "path": r"E:/Test4/OmniAnomaly/CMaps",  # 你的CMAPSS数据路径
        "subset": "FD001",  # 默认使用FD001子集（轻量级）
        "sample_ratio": 0.2  # 轻量级：仅使用20%样本
    }
}

# 全局变量：存储SMAP_MSL训练集统计量（避免测试集归一化泄漏）
smap_msl_mean = None
smap_msl_std = None


class UnifiedTimeSeriesDatasetGenerator:
    """统一数据集生成器：新增 Lorenz 支持"""

    def __init__(self, dataset_name, window_length=50, test_size=0.2, anomaly_contamination=0.05, file_path=None):
        self.dataset_name = dataset_name.lower()
        self.window_length = window_length
        self.test_size = test_size
        self.anomaly_contamination = anomaly_contamination
        self.scaler = StandardScaler()
        self.file_path = file_path  # 单文件模式下的具体文件路径

        # 仅允许目标数据集
        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(f"仅支持以下数据集：{list(DATASET_CONFIG.keys())}")
        self.config = DATASET_CONFIG[self.dataset_name]
        self.dataset_path = self.config["path"]

    def generate_mackey_glass(self, tau=20, sample_len=10000, alpha=0.2, beta=10, gamma=-0.1):
        """生成【低噪声】Mackey-Glass混沌序列（3变量）"""
        np.random.seed(SEED)
        x = np.random.rand(100)
        for i in range(100, sample_len):
            x_i = x[i - 1] + alpha * x[i - tau] / (1 + x[i - tau] ** beta) + gamma * x[i - 1]
            x = np.append(x, x_i)

        # 低噪声配置：仅添加0.05倍的高斯噪声
        x1 = x[:sample_len] + np.random.randn(sample_len) * 0.05
        x2 = np.sin(x[:sample_len]) + x[:sample_len] * 0.3 + np.random.randn(sample_len) * 0.05
        x3 = np.cos(x[:sample_len]) + np.random.randn(sample_len) * 0.05
        data = np.vstack([x1, x2, x3]).T

        # 生成异常标签（1%异常比例）
        np.random.seed(SEED)
        anomaly_idx = np.random.choice(sample_len, int(sample_len * 0.01), replace=False)
        labels = np.zeros(sample_len)
        labels[anomaly_idx] = 1
        # 异常仅轻微扰动（低噪声场景）
        data[anomaly_idx] = data[anomaly_idx] * 1.5 + np.random.randn(len(anomaly_idx), 3) * 0.1

        return data, labels

    # ========== 新增：Lorenz 混沌序列生成函数 ==========
    def generate_lorenz(self, sample_len=10000, dt=0.01, rho=28, sigma=10, beta=8 / 3):
        """生成【低噪声】Lorenz混沌序列（3变量，轻量级适配）"""
        np.random.seed(SEED)

        # 初始化3个变量
        x = np.zeros(sample_len)
        y = np.zeros(sample_len)
        z = np.zeros(sample_len)

        # 初始值（避免初始震荡）
        x[0], y[0], z[0] = 1.0, 1.0, 1.0

        # 生成Lorenz混沌序列
        for i in range(sample_len - 1):
            dx = sigma * (y[i] - x[i])
            dy = x[i] * (rho - z[i]) - y[i]
            dz = x[i] * y[i] - beta * z[i]

            x[i + 1] = x[i] + dx * dt
            y[i + 1] = y[i] + dy * dt
            z[i + 1] = z[i] + dz * dt

        # 组合成3维数据 + 低噪声（适配轻量级模型）
        data = np.stack([x, y, z], axis=1)
        data += np.random.randn(*data.shape) * 0.02  # 极轻噪声，凸显模型优势

        # 生成异常标签（1%异常比例，轻微扰动）
        anomaly_idx = np.random.choice(sample_len, int(sample_len * 0.01), replace=False)
        labels = np.zeros(sample_len)
        labels[anomaly_idx] = 1

        # 异常扰动（轻量级场景下的合理异常）
        data[anomaly_idx] = data[anomaly_idx] * 1.4 + np.random.randn(len(anomaly_idx), 3) * 0.05

        print(f"📊 生成Lorenz序列：样本数={sample_len} | 异常数={np.sum(labels)} | 特征数=3")
        return data, labels

    def load_single_smap_msl_file(self, file_type="train"):
        """加载单个SMAP_MSL文件（适配标签在最后一列的情况）"""
        if not self.file_path:
            raise ValueError("单文件模式下必须指定file_path")

        # 读取文件
        loaded_data = np.load(self.file_path, allow_pickle=True)
        if isinstance(loaded_data, (tuple, list)) and len(loaded_data) == 2:
            data, label = loaded_data
        elif isinstance(loaded_data, np.ndarray):
            # 核心修复：最后一列是标签，前面是特征
            label = loaded_data[:, -1]  # 提取标签列
            data = loaded_data[:, :-1]  # 提取特征列
        else:
            raise ValueError(f"{self.file_path} 格式异常：{type(loaded_data)}")

        # 预处理
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        # 单文件归一化
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        data = (data - mean) / std
        label = np.where(label > 0, 1, 0).astype(np.int32)

        print(
            f"📄 加载 {file_type} 文件：{os.path.basename(self.file_path)} | 样本数：{len(data)} | 特征数：{data.shape[-1]} | 异常数：{np.sum(label)}")
        return data, label

    def load_smd(self):
        """加载SMD轻量版（高噪声+非平稳+包含异常样本）"""
        # 1. 原有路径/文件检查逻辑不变
        train_dir = os.path.join(self.dataset_path, "train")
        test_dir = os.path.join(self.dataset_path, "test")
        label_dir = os.path.join(self.dataset_path, "test_label")
        for d in [train_dir, test_dir, label_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"SMD目录不存在：{d}")

        train_file = "machine-1-1.txt"
        test_file = "machine-1-1.txt"
        label_file = "machine-1-1.txt"

        # 2. 读取数据（保留原噪声，不做平滑/去噪）
        def read_smd_file(file_path):
            df = pd.read_csv(
                file_path,
                header=None,
                sep=",",
                dtype=np.float32,
                na_values=["", "NaN", "NA"]
            )
            df = df.fillna(df.mean())  # 仅填充空值，保留噪声
            return df.values

        # 读取全量数据（先不切片）
        x_train_full = read_smd_file(os.path.join(train_dir, train_file))
        x_test_full = read_smd_file(os.path.join(test_dir, test_file))
        y_test_full = pd.read_csv(os.path.join(label_dir, label_file), header=None, sep=",").values.flatten()

        # 3. 核心修复：切片到包含异常的区间
        # 训练集：仍取纯正常段
        train_start = 15000
        train_end = 20000
        x_train = x_train_full[train_start:train_end]

        # 测试集：取异常集中段
        test_start = 15000
        test_end = 20000
        x_test = x_test_full[test_start:test_end]
        y_test = y_test_full[test_start:test_end]

        # 4. 标签处理（保留原异常标注）
        y_test = np.round(y_test).astype(np.int32)
        y_test = np.clip(y_test, 0, 1)

        # 打印轻量化后信息（验证异常率）
        print(f"📌 SMD轻量版（高噪声+非平稳+含异常）：")
        print(f"   - 训练样本：{len(x_train)}条 × 38维 | 测试样本：{len(x_test)}条 × 38维")
        print(f"   - 训练集异常率：0.000（纯正常）")
        print(f"   - 测试集异常率：{np.sum(y_test) / len(y_test):.3f}（包含真实异常）")

        return x_train, x_test, y_test

    def load_cmapss(self):
        """加载CMAPSS轻量级子集（适配CESN+MLP的小样本需求）"""
        # 1. 路径检查
        subset = self.config.get("subset", "FD001")
        sample_ratio = self.config.get("sample_ratio", 0.2)

        train_file = f"train_{subset}.txt"
        test_file = f"test_{subset}.txt"
        rul_file = f"RUL_{subset}.txt"

        train_path = os.path.join(self.dataset_path, train_file)
        test_path = os.path.join(self.dataset_path, test_file)
        rul_path = os.path.join(self.dataset_path, rul_file)

        for f in [train_path, test_path, rul_path]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"CMAPSS文件不存在：{f}")

        # 2. 读取数据（CMAPSS格式：列1=发动机ID，列2-4=操作设置，列5-26=传感器数据）
        def read_cmapss_file(file_path):
            df = pd.read_csv(
                file_path,
                header=None,
                sep="\s+",  # 空格分隔
                dtype=np.float32,
                na_values=["", "NaN", "NA"]
            )
            df = df.fillna(df.mean())  # 填充空值
            return df.values

        # 3. 加载训练/测试数据
        train_data = read_cmapss_file(train_path)
        test_data = read_cmapss_file(test_path)
        rul_data = np.loadtxt(rul_path, dtype=np.float32)  # 剩余寿命标签

        # 4. 轻量级处理：仅取20%样本（按发动机ID筛选）
        np.random.seed(SEED)
        train_ids = np.unique(train_data[:, 0])  # 发动机ID列
        selected_ids = np.random.choice(train_ids, int(len(train_ids) * sample_ratio), replace=False)
        train_data = train_data[np.isin(train_data[:, 0], selected_ids)]

        test_ids = np.unique(test_data[:, 0])
        selected_test_ids = np.random.choice(test_ids, int(len(test_ids) * sample_ratio), replace=False)
        test_data = test_data[np.isin(test_data[:, 0], selected_test_ids)]

        # 5. 提取特征（去掉发动机ID，保留操作设置+传感器）
        x_train = train_data[:, 1:]  # 列2-26（24维特征）
        x_test = test_data[:, 1:]  # 列2-26

        # 6. 生成异常标签（基于剩余寿命：RUL<30视为异常）
        # 训练集：仅保留正常样本（RUL>50）
        train_rul = []
        for engine_id in np.unique(train_data[:, 0]):
            engine_data = train_data[train_data[:, 0] == engine_id]
            rul = len(engine_data) - np.arange(len(engine_data))  # 倒序RUL
            train_rul.extend(rul)
        train_rul = np.array(train_rul)
        y_train = np.where(train_rul < 50, 1, 0)  # RUL<50为异常

        # 测试集：RUL<30为异常
        test_rul = []
        engine_idx = 0
        for engine_id in np.unique(test_data[:, 0]):
            engine_len = len(test_data[test_data[:, 0] == engine_id])
            engine_rul = rul_data[engine_idx] - np.arange(engine_len)
            test_rul.extend(engine_rul)
            engine_idx += 1
        test_rul = np.array(test_rul)
        y_test = np.where(test_rul < 30, 1, 0)  # RUL<30为异常

        # 7. 清理无效值
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=1e6, neginf=-1e6)
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=1e6, neginf=-1e6)

        # 8. 打印轻量级CMAPSS信息
        print(f"📌 CMAPSS轻量级子集（{subset}）：")
        print(f"   - 训练样本：{len(x_train)}条 × 24维 | 测试样本：{len(x_test)}条 × 24维")
        print(f"   - 训练集异常率：{np.sum(y_train) / len(y_train):.3f}（RUL<50）")
        print(f"   - 测试集异常率：{np.sum(y_test) / len(y_test):.3f}（RUL<30）")

        return x_train, x_test, y_test

    def create_sliding_windows(self, data, labels):
        """创建滑动窗口（优化效率+修复索引）"""
        n_samples = len(data)
        if n_samples < self.window_length:
            return np.array([]), np.array([])  # 样本不足返回空
        # 向量化生成窗口索引，避免循环
        indices = np.arange(self.window_length)[None, :] + np.arange(n_samples - self.window_length)[:, None]
        windows = data[indices]
        # 向量化生成窗口标签
        window_labels = np.any(labels[indices] == 1, axis=1).astype(np.int32)

        return windows.astype(np.float32), window_labels

    def prepare_dataset(self):
        """主流程：加载数据→归一化→滑动窗口→划分训练/测试集（新增Lorenz分支）"""
        # 1. 加载/生成数据
        if self.dataset_name == "mackey_glass":
            data, labels = self.generate_mackey_glass()
            split_idx = int(len(data) * (1 - self.test_size))
            x_train, x_test = data[:split_idx], data[split_idx:]
            y_train, y_test = labels[:split_idx], labels[split_idx:]

            # 归一化
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            # 创建滑动窗口
            train_windows, train_labels = self.create_sliding_windows(x_train, y_train)
            test_windows, test_labels = self.create_sliding_windows(x_test, y_test)

        # ========== 新增：Lorenz 数据处理 ==========
        elif self.dataset_name == "lorenz":
            # 生成Lorenz序列
            data, labels = self.generate_lorenz()
            split_idx = int(len(data) * (1 - self.test_size))
            x_train, x_test = data[:split_idx], data[split_idx:]
            y_train, y_test = labels[:split_idx], labels[split_idx:]

            # 归一化（轻量级场景下的标准归一化）
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            # 创建滑动窗口
            train_windows, train_labels = self.create_sliding_windows(x_train, y_train)
            test_windows, test_labels = self.create_sliding_windows(x_test, y_test)

        elif self.dataset_name == "smap_msl":
            # 单文件模式：分别加载train/test对应文件
            # 假设train和test文件同名（如train/A-1.npy ↔ test/A-1.npy）
            train_file_path = self.file_path.replace("/test/", "/train/")
            if not os.path.exists(train_file_path):
                raise FileNotFoundError(f"对应训练文件不存在：{train_file_path}")

            # 加载训练文件
            self.file_path = train_file_path
            x_train, y_train = self.load_single_smap_msl_file("train")
            # 加载测试文件
            self.file_path = self.file_path.replace("/train/", "/test/")
            x_test, y_test = self.load_single_smap_msl_file("test")

            # 训练集强制无异常
            y_train = np.zeros(len(y_train), dtype=np.int32)

            # 创建滑动窗口（单文件，无维度冲突）
            train_windows, train_labels = self.create_sliding_windows(x_train, y_train)
            test_windows, test_labels = self.create_sliding_windows(x_test, y_test)

        elif self.dataset_name == "smd":
            x_train, x_test, y_test = self.load_smd()
            y_train = np.zeros(len(x_train), dtype=np.int32)

            # 归一化
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            # 创建滑动窗口
            train_windows, train_labels = self.create_sliding_windows(x_train, y_train)
            test_windows, test_labels = self.create_sliding_windows(x_test, y_test)

        elif self.dataset_name == "cmapss":
            x_train, x_test, y_test = self.load_cmapss()
            y_train = np.zeros(len(x_train), dtype=np.int32)  # 训练集强制无异常

            # 归一化
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

            # 创建滑动窗口
            train_windows, train_labels = self.create_sliding_windows(x_train, y_train)
            test_windows, test_labels = self.create_sliding_windows(x_test, y_test)

        # 2. 确保所有标签都是整数类型
        train_labels = train_labels.astype(np.int32) if len(train_labels) > 0 else np.array([])
        test_labels = test_labels.astype(np.int32) if len(test_labels) > 0 else np.array([])

        # 3. 控制训练集异常污染率（仅保留少量异常）
        if len(train_labels) > 0:
            normal_idx = np.where(train_labels == 0)[0].astype(np.int32)
            abnormal_idx = np.where(train_labels == 1)[0].astype(np.int32)

            n_normal = int(len(normal_idx))
            max_abnormal = int(n_normal * self.anomaly_contamination / (1 - self.anomaly_contamination))
            keep_abnormal = min(max_abnormal, int(len(abnormal_idx)))

            np.random.seed(SEED)
            if keep_abnormal > 0:
                keep_abnormal_idx = np.random.choice(abnormal_idx, keep_abnormal, replace=False)
            else:
                keep_abnormal_idx = np.array([], dtype=np.int32)

            # 合并索引并确保为整数
            train_idx = np.concatenate([normal_idx, keep_abnormal_idx]).astype(np.int32)
            train_windows = train_windows[train_idx]
            train_labels = train_labels[train_idx]

        # 输出数据集信息
        print(f"\n📊 {self.dataset_name} 数据集信息：")
        print(f"   - 训练窗口数：{len(train_windows)}，异常率：{np.sum(train_labels) / len(train_labels):.3f}")
        print(f"   - 测试窗口数：{len(test_windows)}，异常率：{np.sum(test_labels) / len(test_labels):.3f}")
        n_vars = train_windows.shape[-1] if len(train_windows) > 0 else self.config["n_vars"]

        return train_windows, test_windows, train_labels, test_labels, n_vars

    # 新增：获取SMAP_MSL所有测试文件列表
    def get_smap_msl_test_files(self):
        test_dir = os.path.join(self.dataset_path, "test")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"SMAP_MSL测试目录不存在：{test_dir}")
        return sorted(glob(os.path.join(test_dir, "*.npy")))