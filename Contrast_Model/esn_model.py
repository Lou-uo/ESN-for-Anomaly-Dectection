import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
import warnings

warnings.filterwarnings('ignore')


class TraditionalESN:
    """
    传统回声状态网络（ESN）实现
    适配时序异常检测场景，无外部依赖，可直接调用
    """

    def __init__(self,
                 n_reservoir=100,
                 spectral_radius=0.95,
                 leaking_rate=0.2,
                 input_scaling=1.0,
                 noise_level=0.01,
                 random_state=42):
        # 核心参数
        self.n_reservoir = n_reservoir  # 储备池神经元数量
        self.spectral_radius = spectral_radius  # 谱半径
        self.leaking_rate = leaking_rate  # 泄漏率
        self.input_scaling = input_scaling  # 输入缩放
        self.noise_level = noise_level  # 噪声水平
        self.random_state = random_state

        # 内部变量
        self.W_in = None  # 输入权重矩阵
        self.W = None  # 储备池权重矩阵
        self.W_out = None  # 输出权重矩阵
        self.scaler = StandardScaler()  # 内置标准化器
        self.state = None  # 储备池状态

        # 初始化随机种子
        np.random.seed(self.random_state)

    def _init_weights(self, n_inputs):
        """初始化输入和储备池权重矩阵"""
        # 1. 输入权重矩阵: [n_reservoir, n_inputs + 1] (包含偏置项)
        self.W_in = (np.random.rand(self.n_reservoir, n_inputs + 1) - 0.5) * self.input_scaling

        # 2. 储备池权重矩阵: [n_reservoir, n_reservoir]
        self.W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5

        # 调整谱半径
        eigenvalues = np.linalg.eigvals(self.W)
        self.W = self.W / (np.max(np.abs(eigenvalues)) / self.spectral_radius)

        # 初始化储备池状态
        self.state = np.zeros(self.n_reservoir)

    def _update_state(self, input_vec):
        """更新储备池状态"""
        # 添加偏置项
        input_with_bias = np.hstack([input_vec, 1.0])

        # 计算新状态
        pre_activation = np.dot(self.W_in, input_with_bias) + np.dot(self.W, self.state)
        # 非线性激活（tanh）+ 噪声
        new_state = np.tanh(pre_activation) + self.noise_level * (np.random.rand(self.n_reservoir) - 0.5)
        # 泄漏积分
        self.state = (1 - self.leaking_rate) * self.state + self.leaking_rate * new_state

        return self.state

    def train(self, train_windows):
        """
        训练ESN模型
        :param train_windows: 训练窗口数据，形状 [n_windows, window_length, n_features]
        """
        # 展平数据（ESN处理时序的方式：每个窗口最后一个时间步作为预测目标）
        n_windows, window_length, n_features = train_windows.shape

        # 标准化（内置逻辑，无需外部类）
        train_flat = train_windows.reshape(-1, n_features)
        train_flat_scaled = self.scaler.fit_transform(train_flat)
        train_windows_scaled = train_flat_scaled.reshape(n_windows, window_length, n_features)

        # 初始化权重
        self._init_weights(n_features)

        # 收集储备池状态和目标
        states_matrix = []
        targets = []

        print(f"📈 开始训练ESN | 储备池大小：{self.n_reservoir} | 训练窗口数：{n_windows}")

        for window in train_windows_scaled:
            # 重置状态
            self.state = np.zeros(self.n_reservoir)

            # 遍历窗口内的每个时间步
            for t in range(window_length):
                input_vec = window[t]
                self._update_state(input_vec)

            # 收集最后一个时间步的状态和目标（重构任务）
            states_matrix.append(self.state)
            targets.append(window[-1])  # 目标是重构最后一个时间步

        # 转换为矩阵
        states_matrix = np.array(states_matrix)
        targets = np.array(targets)

        # 训练输出权重（岭回归，避免过拟合）
        regularization = 1e-6
        self.W_out = np.dot(
            pinv(states_matrix.T @ states_matrix + regularization * np.eye(self.n_reservoir)),
            states_matrix.T @ targets
        )

        print("✅ ESN训练完成")

    def predict(self, test_windows):
        """
        预测异常分数
        :param test_windows: 测试窗口数据，形状 [n_windows, window_length, n_features]
        :return: anomaly_scores: 异常分数，形状 [n_windows]
        """
        n_windows, window_length, n_features = test_windows.shape

        # 标准化
        test_flat = test_windows.reshape(-1, n_features)
        test_flat_scaled = self.scaler.transform(test_flat)
        test_windows_scaled = test_flat_scaled.reshape(n_windows, window_length, n_features)

        anomaly_scores = []

        print(f"🔍 开始预测ESN | 测试窗口数：{n_windows}")

        for window in test_windows_scaled:
            # 重置状态
            self.state = np.zeros(self.n_reservoir)

            # 遍历窗口内的每个时间步
            for t in range(window_length):
                input_vec = window[t]
                self._update_state(input_vec)

            # 预测最后一个时间步
            pred = np.dot(self.state, self.W_out)
            # 计算重构误差（异常分数）
            error = np.mean((pred - window[-1]) ** 2)
            anomaly_scores.append(error)

        # 归一化异常分数到 [0, 1]
        anomaly_scores = np.array(anomaly_scores)
        anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (
                    np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-8)

        return anomaly_scores


# 测试代码（可选，验证模型是否能独立运行）
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.rand(100, 50, 3)  # 100个窗口，每个窗口50步，3个特征
    test_labels = np.random.randint(0, 2, 100)

    # 初始化并训练ESN
    esn = TraditionalESN(n_reservoir=50, spectral_radius=0.9, leaking_rate=0.1)
    esn.train(test_data)

    # 预测
    scores = esn.predict(test_data)
    print(f"✅ 测试完成 | 异常分数形状：{scores.shape} | 分数范围：[{np.min(scores):.4f}, {np.max(scores):.4f}]")