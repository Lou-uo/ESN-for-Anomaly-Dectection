​	Time series anomaly detection plays a critical role in industrial monitoring, fault diagnosis, and predictive maintenance. Over the past decades, numerous methods have been proposed, which can be broadly categorized into traditional statistical methods, distance- and density-based methods, deep learning-based methods, and reservoir computing methods such as echo state networks (ESNs). 

​	时间序列异常检测在工业监控、故障诊断和预测维护中起着至关重要的作用。在过去的几十年中，已经提出了许多方法，这些方法可以大致分为传统的统计方法、基于距离和密度的方法、基于深度学习的方法和储层计算方法，如回声状态网络(ESNs)。

​	Traditional methods rely on statistical assumptions and handcrafted rules. Typical methods include OCSVM,  and Isolation Forest (IForest). These methods are computationally efficient but fail to capture nonlinear temporal dependencies and perform poorly on complex industrial data with noise, non-stationarity, and distribution drift. 

​	传统方法依赖于统计假设和手工规则。典型的方法包括OCSVM和孤立森林(IForest)。这些方法在计算上是高效的，但是不能捕捉非线性时间依赖性，并且在具有噪声、非平稳性和分布漂移的复杂工业数据上表现不佳。

​	Deep learning models have been widely studied due to their powerful feature extraction capability. Autoencoder (AE), LSTM, GRU, DAGMM, and USAD learn normality via reconstruction or prediction. Recently, advanced models such as Anomaly Transformer, CrossAD achieve state-of-the-art performance by using multi-scale or attention mechanisms. However, these models are computationally intensive, difficult to deploy in online scenarios, and lack robustness under strong noise and outliers.

​	深度学习模型由于其强大的特征提取能力而得到了广泛的研究。自动编码器(AE)、LSTM、GRU和USAD通过重建或预测学习常态。最近，Anomaly Transformer、CrossAD等高级模型通过使用多尺度或注意力机制实现了最先进的性能。然而，这些模型是计算密集型的，难以在在线场景中部署，并且在强噪声和异常值下缺乏鲁棒性。

​	 Echo State Networks (ESNs) are lightweight recurrent models suitable for real-time time series tasks. Benefiting from the fixed random reservoir, ESNs have low training cost and high efficiency. Recent variants such as ESN-AE and AESN improve representation ability, but still suffer from noise sensitivity, distribution shift, and limited capability in uncertainty modeling. According to our extensive research, no existing ESN-based method integrates correlation entropy loss, reversible normalization, and uncertainty-aware learning for robust anomaly detection. 

​	回声状态网络是一种轻量级递归模型，适用于实时时间序列任务。得益于固定的随机库，进化神经网络训练成本低，效率高。最近的变体如ESN-AE和AESN提高了表示能力，但仍然受到噪声敏感性、分布偏移和不确定性建模能力有限的影响。据我们所知，没有现有的基于ESN的方法集成了相关熵损失、可逆归一化和不确定性感知学习，用于鲁棒的异常检测。

​	Most previous works adopt point-level metrics including Precision, Recall, F1-score, AUC-ROC, and AUC-PR. Recent studies have demonstrated that VUS-ROC and VUS-PR are more robust and fair for temporal localization. In this work, we adopt these metrics for comprehensive evaluation. 

​	大多数以前的工作采用点级度量，包括精确度、召回率、F1分数、AUC-ROC和AUC-PR。最近的研究表明，VUS-ROC和VUS-PR对时间定位更稳健和公平。在这项工作中，我们采用这些指标进行综合评价。

​	In summary, existing time series anomaly detection methods still face several challenges: 1) poor robustness to noise and outliers; 2) performance degradation under distribution shift; 3) lack of uncertainty perception for industrial anomalies. To overcome these limitations, we propose the Correntropy-based Echo State Network (CESN) for multivariate time series anomaly detection.

​	总之，现有时序异常检测方法仍然面临几个挑战: 1)对噪声和异常值的鲁棒性差；2)分布变化下的性能下降；3)缺乏对产业异常的不确定性感知。为了克服这些局限性，我们提出了基于相关熵的回声状态网络(CESN)用于多元时间序列异常检测。