import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')


class IFAnomalyDetector:
    """
    基于孤立森林（Isolation Forest）的时序异常检测器
    适配时序窗口数据（需展平为二维特征）
    """

    def __init__(self, n_estimators=100, contamination=0.05, max_samples="auto",
                 max_features=1.0, bootstrap=False, random_state=42):
        """
        初始化孤立森林检测器
        :param n_estimators: 决策树数量，默认100
        :param contamination: 异常比例（与数据污染率对齐）
        :param max_samples: 构建每棵树使用的样本数，"auto"=min(256, n_samples)
        :param max_features: 构建每棵树使用的特征数比例，默认1.0（全部特征）
        :param bootstrap: 是否使用bootstrap抽样，默认False
        :param random_state: 随机种子（保证可复现）
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.iforest = None
        self.n_features = None  # 记录展平后的特征数

    def train(self, train_windows, verbose=True):
        """
        训练孤立森林模型（时序窗口展平为二维特征）
        :param train_windows: 训练时序窗口，shape=(n_windows, window_length, n_vars)
        :param verbose: 是否打印训练信息
        :return: None
        """
        # 展平时序窗口：(n_windows, window_length * n_vars)
        train_flat = train_windows.reshape(len(train_windows), -1)
        self.n_features = train_flat.shape[1]

        if verbose:
            print(f"📚 IF训练数据展平完成 | 样本数：{len(train_flat)} | 特征数：{self.n_features}")

        # 初始化并训练孤立森林
        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1  # 使用所有CPU核心加速训练
        )
        self.iforest.fit(train_flat)

        if verbose:
            print(f"✅ IF训练完成 | 树数量：{self.n_estimators} | 异常比例：{self.contamination}")

    def predict(self, test_windows):
        """
        预测异常分数（平均路径长度的负值，值越大越异常）
        :param test_windows: 测试时序窗口，shape=(n_windows, window_length, n_vars)
        :return: anomaly_scores: 异常分数数组
        """
        if self.iforest is None:
            raise ValueError("❌ 模型未训练，请先调用train()方法")

        # 展平测试窗口
        test_flat = test_windows.reshape(len(test_windows), -1)

        # IF决策函数：正值=正常，负值=异常 → 转换为异常分数（负值取反）
        decision_scores = self.iforest.decision_function(test_flat)
        anomaly_scores = -decision_scores  # 异常分数：越大越异常

        # 处理极端值（避免评估时出错）
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)

        return anomaly_scores

    def evaluate(self, test_windows, test_labels, verbose=True):
        """
        快速评估模型性能（基础指标）
        :param test_windows: 测试窗口
        :param test_labels: 测试标签
        :param verbose: 是否打印结果
        :return: 评估结果字典
        """
        anomaly_scores = self.predict(test_windows)

        # 计算核心指标
        auroc = roc_auc_score(test_labels, anomaly_scores)
        precision, recall, _ = precision_recall_curve(test_labels, anomaly_scores)
        aupr = auc(recall, precision)

        results = {
            "auroc": auroc,
            "aupr": aupr,
            "anomaly_scores": anomaly_scores
        }

        if verbose:
            print(f"\n📊 IF评估结果：")
            print(f"   AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

        return results


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 模拟时序窗口数据
    np.random.seed(42)
    train_windows = np.random.randn(1000, 20, 5).astype(np.float32)  # (1000个窗口，窗口长度20，5个特征)
    test_windows = np.random.randn(200, 20, 5).astype(np.float32)
    test_labels = np.random.randint(0, 2, size=200)  # 0=正常，1=异常

    # 初始化并训练IF
    detector = IFAnomalyDetector(contamination=0.05, n_estimators=100)
    detector.train(train_windows)

    # 预测和评估
    detector.evaluate(test_windows, test_labels)