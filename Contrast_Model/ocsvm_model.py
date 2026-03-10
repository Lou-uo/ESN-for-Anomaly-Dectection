import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')


class OCSVMAnomalyDetector:
    """
    基于One-Class SVM的时序异常检测器
    适配时序窗口数据（需展平为二维特征）
    """

    def __init__(self, nu=0.05, kernel="rbf", gamma="scale", random_state=42):
        """
        初始化OCSVM检测器
        :param nu: 异常比例的上界（通常设为训练数据中的异常污染率）
        :param kernel: 核函数类型，可选"rbf", "linear", "poly", "sigmoid"
        :param gamma: 核函数参数，"scale"表示1/(n_features*X.var())
        :param random_state: 随机种子（保证可复现）
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.ocsvm = None
        self.n_features = None  # 记录展平后的特征数

    def train(self, train_windows, verbose=True):
        """
        训练OCSVM模型（时序窗口展平为二维特征）
        :param train_windows: 训练时序窗口，shape=(n_windows, window_length, n_vars)
        :param verbose: 是否打印训练信息
        :return: None
        """
        # 展平时序窗口：(n_windows, window_length * n_vars)
        train_flat = train_windows.reshape(len(train_windows), -1)
        self.n_features = train_flat.shape[1]

        if verbose:
            print(f"📚 OCSVM训练数据展平完成 | 样本数：{len(train_flat)} | 特征数：{self.n_features}")

        # 初始化并训练OCSVM
        self.ocsvm = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.ocsvm.fit(train_flat)

        if verbose:
            print(f"✅ OCSVM训练完成 | 核函数：{self.kernel} | gamma：{self.gamma} | nu：{self.nu}")

    def predict(self, test_windows):
        """
        预测异常分数（距离超平面的距离，负值为异常）
        :param test_windows: 测试时序窗口，shape=(n_windows, window_length, n_vars)
        :return: anomaly_scores: 异常分数数组，值越大越异常
        """
        if self.ocsvm is None:
            raise ValueError("❌ 模型未训练，请先调用train()方法")

        # 展平测试窗口
        test_flat = test_windows.reshape(len(test_windows), -1)

        # OCSVM决策函数：正值=正常，负值=异常 → 转换为异常分数（负值取反，值越大越异常）
        decision_scores = self.ocsvm.decision_function(test_flat)
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
            print(f"\n📊 OCSVM评估结果：")
            print(f"   AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

        return results


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 模拟时序窗口数据
    np.random.seed(42)
    train_windows = np.random.randn(1000, 20, 5).astype(np.float32)  # (1000个窗口，窗口长度20，5个特征)
    test_windows = np.random.randn(200, 20, 5).astype(np.float32)
    test_labels = np.random.randint(0, 2, size=200)  # 0=正常，1=异常

    # 初始化并训练OCSVM
    detector = OCSVMAnomalyDetector(nu=0.05, kernel="rbf", gamma="scale")
    detector.train(train_windows)

    # 预测和评估
    detector.evaluate(test_windows, test_labels)