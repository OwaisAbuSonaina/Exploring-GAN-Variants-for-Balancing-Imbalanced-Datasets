# Credit Card Fraud Detection: GAN-based Data Augmentation

A comparative study exploring the effectiveness of Generative Adversarial Networks (Vanilla GAN vs. WGAN) in addressing extreme class imbalance for credit card fraud detection.

## Project Overview

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is highly imbalanced, with fraud accounting for only **0.17%** of transactions. Standard classifiers often fail to detect fraud in such scenarios, favoring the majority class.

This project implements and compares two generative models to synthesize realistic fraud samples:

1. **Vanilla GAN** (Standard Multi-Layer Perceptron GAN)
2. **Wasserstein GAN (WGAN)** (With weight clipping and Wasserstein loss)

The goal is to improve the **Recall** of the classifier (Random Forest) without significantly sacrificing precision.

## Key Features

* **Data Preprocessing:** Robust scaling of Time/Amount features; PCA features left as-is.
* **GAN Implementation:** Custom PyTorch implementations of Generator, Discriminator (Vanilla), and Critic (WGAN) tailored for tabular data.
* **Stability Techniques:** Implemented Wasserstein Loss and Weight Clipping to prevent mode collapse.
* **Evaluation:** Comprehensive analysis using Precision-Recall Curves, F1-Scores, and Confusion Matrices.

## Results & Analysis

We compared three scenarios:

1. **Baseline:** Random Forest trained on the original imbalanced data.
2. **Vanilla GAN:** Augmented with synthetic fraud samples from a standard GAN.
3. **WGAN:** Augmented with synthetic samples from a Wasserstein GAN.

| Model | Recall (Fraud Caught) | Precision (False Alarms) | F1-Score |
| --- | --- | --- | --- |
| **Baseline** | 0.81 | **0.94** | **0.87** |
| **Vanilla GAN** | 0.83 | 0.83 | 0.83 |
| **WGAN** | **0.89** | 0.43 | 0.58 |

**Key Finding:** The **WGAN** model successfully forced the classifier to be more aggressive, achieving the highest Recall (catching the most fraud). However, this came at the cost of higher False Positives, highlighting the classic Precision-Recall trade-off in fraud detection.

*Figure 1: Comparison of Confusion Matrices across all three scenarios.*

## Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-gan.git
cd fraud-detection-gan

```


2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch tqdm

```


3. **Run the Notebook**
Open `Main.ipynb` in Jupyter Notebook or Google Colab to reproduce the training and evaluation pipeline.

## Project Structure

```
├── models/               # Saved PyTorch models (Generator/Critic weights)
├── images/               # Plots and visualizaions
├── Main.ipynb            # Core logic (Preprocessing -> Training -> Evaluation)
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies

```

## Future Work

* Implement **WGAN-GP** (Gradient Penalty) to improve stability over simple weight clipping.
* Experiment with **SMOTE** to compare GAN-based augmentation vs. traditional oversampling.
* Tune the Random Forest decision threshold to optimize the F1-score for the WGAN scenario.

---
