# Credit Card Fraud Detection: GAN-based Data Augmentation

A comparative study exploring the effectiveness of Generative Adversarial Networks (Vanilla GAN vs. WGAN) in addressing extreme class imbalance for credit card fraud detection.

## 📌 Project Overview

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is highly imbalanced, with fraud accounting for only **0.17%** of transactions. Standard classifiers often fail to detect fraud in such scenarios, favoring the majority class.

This project implements and compares two generative models to synthesize realistic fraud samples:

1. **Vanilla GAN** (Standard Multi-Layer Perceptron GAN)
2. **Wasserstein GAN (WGAN)** (With weight clipping and Wasserstein loss)

The goal is to improve the **Recall** of the classifier (Random Forest) without significantly sacrificing precision.

## 📊 Visual Analysis

### 1. The Challenge: Extreme Imbalance

The dataset contains a massive disparity between normal and fraudulent transactions. This imbalance causes standard models to bias heavily toward the "Normal" class.

*Figure 1: Visual representation of the 99.8% vs 0.2% class distribution.*

### 2. The Solution: GAN-based Augmentation

We trained three Random Forest classifiers under different conditions. The Confusion Matrices below reveal how GAN augmentation shifts the model's behavior.

*Figure 2: Comparison of Confusion Matrices. Note the shift in False Positives and True Positives in the WGAN scenario.*

**Key Observations:**

* **Baseline (Scenario A):** High Precision, but misses 18 fraud cases.
* **WGAN (Scenario C):** Drastically improved Recall (only 10 missed frauds), but at the cost of higher False Positives.

### 3. Performance Trade-off

The WGAN model successfully forces the classifier to be more aggressive. As shown below, while Precision drops, the WGAN achieves the highest **Recall**, which is often the priority in fraud detection (catching the thief is more important than annoying a customer).

*Figure 3: Precision vs. Recall trade-off across the three scenarios.*

*Figure 4: Precision-Recall Curve demonstrating the stability of the models at different thresholds.*

## 📈 Quantitative Results

| Model | Recall (Fraud Caught) | Precision (False Alarms) | F1-Score |
| --- | --- | --- | --- |
| **Baseline** | 0.81 | **0.94** | **0.87** |
| **Vanilla GAN** | 0.83 | 0.83 | 0.83 |
| **WGAN** | **0.89** | 0.43 | 0.58 |

## 🛠️ Key Features

* **Data Preprocessing:** Robust scaling of Time/Amount features; PCA features left as-is.
* **GAN Implementation:** Custom PyTorch implementations of Generator, Discriminator (Vanilla), and Critic (WGAN) tailored for tabular data.
* **Stability Techniques:** Implemented Wasserstein Loss and Weight Clipping to prevent mode collapse.

## 🚀 Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-gan.git
cd fraud-detection-gan

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Run the Notebook**
Open `Main.ipynb` in Jupyter Notebook or Google Colab to reproduce the training and evaluation pipeline.

## 📂 Project Structure

```
├── models/               # Saved PyTorch models (Generator/Critic weights)
├── images/               # Project visualizations
│   ├── class_imbalance.png
│   ├── confusion_matrices.png
│   ├── metrics_tradeoff.png
│   └── pr_curve.png
├── Main.ipynb            # Core logic (Preprocessing -> Training -> Evaluation)
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies

```

---
