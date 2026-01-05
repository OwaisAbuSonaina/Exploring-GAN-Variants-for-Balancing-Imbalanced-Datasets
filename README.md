# Credit Card Fraud Detection: GAN-based Data Augmentation

A comparative study exploring the effectiveness of Generative Adversarial Networks (Vanilla GAN vs. WGAN) in addressing extreme class imbalance for credit card fraud detection.

## Project Overview

The **Credit Card Fraud Detection** dataset is highly imbalanced, where fraudulent transactions represent only **0.17%** of the data. Traditional classifiers (like Random Forest) often biasedly predict "Normal" for everything, achieving high accuracy but missing actual fraud cases.

This project implements two generative models to synthesize realistic fraud samples and balance the dataset:

1. **Vanilla GAN** (Standard MLP GAN)
2. **Wasserstein GAN (WGAN)** (With weight clipping and Wasserstein loss)

## 📊 Dataset & Imbalance Analysis

The dataset contains 284,807 transactions, but only 492 are fraudulent. This extreme imbalance poses a significant challenge for supervised learning.

*Figure 1: The extreme class imbalance (0.2% vs 99.8%) visualized.*

## 🛠️ Methodology

We designed a 3-stage pipeline:

1. **Preprocessing:** Robust scaling of 'Time' and 'Amount' features.
2. **Data Augmentation:** Trained GANs on the minority class to generate synthetic fraud samples.
3. **Classification:** Trained a Random Forest classifier on three distinct scenarios:
* **Scenario A:** Original Imbalanced Data (Baseline)
* **Scenario B:** Augmented with Vanilla GAN Data
* **Scenario C:** Augmented with WGAN Data



## Results & Analysis

### 1. Confusion Matrix Comparison

The WGAN model significantly altered the decision boundary, making the classifier more sensitive to fraud.

*Figure 2: Confusion Matrices for Baseline, Vanilla GAN, and WGAN scenarios.*

### 2. Performance Metrics

While the Baseline model achieved the highest Precision, the **WGAN model achieved the highest Recall (88 captured frauds)**, proving it is the most effective at minimizing false negatives (missed fraud).

| Model | Recall (Fraud Caught) | Precision (False Alarms) | F1-Score |
| --- | --- | --- | --- |
| **Baseline** | 0.81 | **0.94** | **0.87** |
| **Vanilla GAN** | 0.83 | 0.83 | 0.83 |
| **WGAN** | **0.89** | 0.43 | 0.58 |

*Figure 3: The trade-off between Precision and Recall across the three models.*

### 3. Precision-Recall Curve

The curve below demonstrates that the WGAN model is not "worse" but rather "more aggressive." Adjusting the classification threshold could optimize the F1-score.

*Figure 4: Precision-Recall Curves showing model stability.*

## Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-gan.git
cd fraud-detection-gan

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Run the Analysis**
Open `Main.ipynb` in Jupyter Notebook to reproduce the training and evaluation pipeline.

## Project Structure

```
├── models/               # Saved PyTorch models (Generator/Critic weights)
├── images/               # Visualization plots (Confusion Matrix, PR Curve, etc.)
├── Main.ipynb            # Core logic (Preprocessing -> Training -> Evaluation)
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies

```

## Future Work

* Implement **WGAN-GP** (Gradient Penalty) to improve training stability.
* Perform hyperparameter tuning on the Random Forest threshold to reduce WGAN False Positives.
* Deploy the model using **FastAPI** with a dedicated inference endpoint.

---
