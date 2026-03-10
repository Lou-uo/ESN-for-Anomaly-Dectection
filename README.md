# ESN-for-Anomaly-Detection

## 📌 Project Overview

This project focuses on **multivariate time-series anomaly detection in industrial scenarios**.  
The goal is to design a **lightweight, robust, and uncertainty-aware anomaly detection model** suitable for **edge deployment**.

Traditional methods suffer from three major limitations:

1. **Poor scale robustness**  
   Industrial sensor data (e.g., C-MAPSS, SMD) often contain large magnitude differences across variables and temporal distribution shifts.  
   Conventional normalization methods may lead to missed anomalies.

2. **Lack of uncertainty awareness**  
   Most RNN-based models (e.g., GRU/LSTM) produce deterministic predictions, which cannot quantify prediction uncertainty and struggle with point anomalies.

3. **Edge deployment challenges**  
   Deep models such as GRU/LSTM/Transformer have large parameter sizes and high inference latency, making them unsuitable for resource-constrained edge devices (e.g., Raspberry Pi, industrial gateways).

### 🎯 Final Objective

Design a **lightweight anomaly detection framework** that:

- maintains strong detection performance
- provides **uncertainty estimation**
- supports **edge deployment**

---

# 🧠 Methodology

The proposed framework is based on a **Chaotic Echo State Network (CESN)** architecture with four key modules.

## 1️⃣ Lightweight ESN Backbone

We adopt **Echo State Network (ESN)** as the core architecture.

Key characteristics:

- Reservoir weights are fixed
- Only the readout layer is trained

Advantages:

- Parameter count **< 500k**
- About **1/10 the size of GRU**
- Avoids gradient vanishing
- Training efficiency **10× faster**

---

## 2️⃣ Scale Robustness: ReVIN Normalization

To address distribution shift and scale differences, we introduce:

**Reversible Instance Normalization (ReVIN)**

Features:

- Instance-level normalization for each time series
- Computes mean and variance per sample
- Supports reversible transformation

Benefits:

- Handles multi-variable scale differences
- Mitigates temporal distribution drift

---

## 3️⃣ Uncertainty Awareness: Stochastic Readout Layer

We introduce **DropConnect** in the ESN readout layer.

Combined with **Monte Carlo sampling**, the model estimates:

- prediction mean
- prediction variance

This enables **uncertainty-aware anomaly detection**.

Observation:

- Anomalous samples show higher prediction variance
- Improves detection of point anomalies (e.g., SMD dataset)

---

## 4️⃣ Structural Anomaly Detection: MCC Loss

We design a custom **MCC Loss**, which combines:

- reconstruction error
- uncertainty term
- geometric constraint

The geometric constraint measures the **norm deviation of reservoir embeddings**, helping detect structural anomalies in time-series patterns.

This addresses a common weakness of pure reconstruction-based models.

---

## 5️⃣ Adaptive Thresholding: POT-EVT

We use **Peak Over Threshold (POT)** based on **Extreme Value Theory (EVT)** to determine anomaly thresholds.

Advantages:

- Automatically adapts to dataset distribution
- Avoids fixed threshold issues
- Reduces false positives and missed detections

---

# 📊 Benchmark Comparison

We compare CESN with several mainstream anomaly detection models.

| Model | Key Characteristics | CESN Advantages |
|------|------|------|
| GRU / LSTM | Gated recurrent models with strong sequence modeling | 90% fewer parameters, 5–10× faster inference, +2% AUROC on SMD |
| Isolation Forest | Lightweight tree-based method | Supports temporal dependency modeling, +5% AUROC on C-MAPSS |
| One-Class SVM | Classical anomaly detection model | More robust for high-dimensional industrial data |
………

---

# 📈 Experimental Results

Experiments are conducted on two industrial benchmark datasets:

- **C-MAPSS** (progressive degradation anomalies)
- **SMD** (point anomalies)

### Detection Performance

| Dataset | AUROC |
|------|------|
| C-MAPSS | **97%+** |
| SMD | **98.5%+** |
………

CESN slightly outperforms GRU-based models in most cases.

---

# ⚡ Deployment Performance

| Metric | CESN |
|------|------|
| Parameter size | < 500k |
| Memory usage | < 5MB |
| Inference latency | 1–10 ms |

The model fully satisfies **industrial real-time edge deployment requirements**.

---

# ✅ Key Advantages

The proposed CESN framework achieves a strong balance between:

- **Detection performance**
- **Lightweight architecture**
- **Robustness to distribution shifts**

Compared with traditional approaches, CESN provides an **efficient solution for multivariate anomaly detection in industrial edge scenarios**.
