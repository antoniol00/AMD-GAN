# AMD-GAN: Class-Specific Adaptive GANs for Configurable Synthetic Traffic and Imbalance Correction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **AMD-GAN**, a novel framework for generating synthetic network traffic using Class-Specific Wasserstein GANs with Gradient Penalty (WGAN-GP). The system is designed to address class imbalance in Network Intrusion Detection Systems (NIDS) by generating high-quality synthetic samples for minority attack classes.

## 📋 Table of Contents

- [AMD-GAN: Class-Specific Adaptive GANs for Configurable Synthetic Traffic and Imbalance Correction](#amd-gan-class-specific-adaptive-gans-for-configurable-synthetic-traffic-and-imbalance-correction)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
    - [Architecture](#architecture)
  - [✨ Key Features](#-key-features)
  - [📁 Repository Structure](#-repository-structure)
  - [🔧 Installation](#-installation)
    - [Requirements](#requirements)
    - [Dependencies](#dependencies)
  - [🚀 Usage](#-usage)
    - [1. Training WGAN-GP Models](#1-training-wgan-gp-models)
    - [2. Generating Synthetic Datasets](#2-generating-synthetic-datasets)
    - [3. Train-Synthetic-Test-Real (TSTR) Evaluation](#3-train-synthetic-test-real-tstr-evaluation)
    - [4. Oversampling Comparison Experiment](#4-oversampling-comparison-experiment)
  - [📄 License](#-license)
  - [🤝 Acknowledgments](#-acknowledgments)

## 🎯 Overview

Network Intrusion Detection Systems suffer from severe class imbalance, where benign traffic vastly outnumbers attack samples. Traditional oversampling techniques (SMOTE, ADASYN) often fail to capture the complex distributions of network traffic data.

**AMD-GAN** proposes a class-specific GAN approach where:
- Each attack class has its own dedicated WGAN-GP model
- Minority classes use adaptive configurations (smaller batch sizes, more epochs, stronger regularization)
- Synthetic data quality is validated through Train-Synthetic-Test-Real (TSTR) protocols

### Architecture

The framework uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** with the following characteristics:
- **Generator**: Multi-layer perceptron with LeakyReLU activations
- **Critic**: Wasserstein distance estimator with gradient penalty
- **Class-Specific Training**: Separate models for each traffic class
- **Adaptive Configuration**: Dynamic hyperparameters based on class size

## ✨ Key Features

- **Class-Specific GANs**: Dedicated WGAN-GP model for each traffic class (BENIGN, DDoS, DoS, Bot, Brute Force, Port Scan, Web Attack)
- **Adaptive Training**: Automatic configuration adjustment based on class sample size
- **Configurable Generation**: Generate custom datasets with specified samples per class
- **Multiple Evaluation Protocols**: TSTR (multiclass/binary), TRTR baselines
- **Comparison Framework**: Benchmark against SMOTE, ADASYN, BorderlineSMOTE, SMOTE-ENN, Random Oversampling

## 📁 Repository Structure

```
AMD-GAN/
│
├── gan_wgan_minority_classes.py      # WGAN-GP training with adaptive configurations
├── generate_synthetic_dataset.py      # Synthetic dataset generation utility
├── train_synthetic_test_real.py    # TSTR/TRTR evaluation framework
├── experiment_oversampling_comparison.py  # Comparison with traditional oversampling
│
├── outputs_wgan_multi/            # Trained WGAN-GP models
│   ├── benign/                       # BENIGN class model
│   ├── bot/                          # Bot class model
│   ├── brute_force/                  # Brute Force class model
│   ├── ddos/                         # DDoS class model
│   ├── dos/                          # DoS class model
│   ├── port_scan/                    # Port Scan class model
│   └── web_attack/                   # Web Attack class model
│
├── results_tstr/                     # TSTR evaluation results
│   ├── completo_v2_balanceado/       # Results for balanced dataset
│   ├── completo_v2_uniforme/         # Results for uniform dataset
│
└── results_oversampling/             # Oversampling comparison results
    └── comparison_*/                 # Timestamped comparison results
```

## 🔧 Installation

### Requirements

```bash
pip install numpy pandas polars scikit-learn tensorflow lightgbm xgboost imbalanced-learn matplotlib scipy
```

### Dependencies

- Python ≥ 3.8
- TensorFlow ≥ 2.x
- NumPy, Pandas, Polars
- Scikit-learn
- LightGBM, XGBoost
- imbalanced-learn (for comparison experiments)
- Matplotlib, Scipy

## 🚀 Usage

### 1. Training WGAN-GP Models

Train class-specific WGAN-GP models for minority classes:

```bash
# Train all minority classes
python gan_wgan_minority_classes.py

# Train specific classes
python gan_wgan_minority_classes.py --classes Bot "Web Attack"

# Train only Bot class
python gan_wgan_minority_classes.py --classes Bot
```

**Configuration tiers based on class size:**

| Class Size | Batch Size | Epochs | Lambda GP | Oversample Factor |
|------------|------------|--------|-----------|-------------------|
| Large (>15k) | 128 | 15,000 | 10.0 | 1x |
| Small (<15k) | 32 | 25,000 | 15.0 | 10x |
| Very Small (<5k) | 16 | 30,000 | 20.0 | 20x |

### 2. Generating Synthetic Datasets

Generate custom synthetic datasets using trained models:

```bash
# Interactive mode
python generate_synthetic_dataset.py --interactive

# Balanced dataset (equal samples per class)
python generate_synthetic_dataset.py --balanced 10000

# Custom distribution
python generate_synthetic_dataset.py --benign 50000 --ddos 20000 --dos 15000 --bot 5000

# Using v2 models (improved for minority classes)
python generate_synthetic_dataset.py --use-v2 --balanced 10000

# From configuration file
python generate_synthetic_dataset.py --config my_config.json
```

### 3. Train-Synthetic-Test-Real (TSTR) Evaluation

Evaluate synthetic data quality using TSTR protocol:

```bash
# Interactive mode - select dataset
python train_synthetic_test_real_v2.py

# Specific dataset
python train_synthetic_test_real_v2.py --dataset completo_v2_balanceado

# List available datasets
python train_synthetic_test_real_v2.py --list
```

**Evaluation scenarios:**
- **TSTR Multiclass**: Train on synthetic, test on real (7 classes)
- **TRTR Multiclass**: Train on real, test on real (baseline)
- **TSTR Binary**: Train on synthetic, test on real (BENIGN vs ATTACK)
- **TRTR Binary**: Train on real, test on real (binary baseline)

### 4. Oversampling Comparison Experiment

Compare GAN-based oversampling with traditional techniques:

```bash
python experiment_oversampling_comparison.py
```

**Compared techniques:**
- Original (no oversampling) - baseline
- **AMD-GAN** (our approach)
- SMOTE
- ADASYN
- BorderlineSMOTE
- SMOTE-ENN
- Random Oversampling


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Dataset: CIC-IDS2017 from the Canadian Institute for Cybersecurity
- This repository is part of a publication, which is also part of the project "CiberIA: Investigación e Innovación para la Integración de Ciberseguridad e Inteligencia Artificial (Proyecto C079/23)", financed by "European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience", through INCIBE. It has also been partially supported by the project SecAI (PID2022-139268OB-I00) funded by the Spanish Ministerio de Ciencia e Innovacion, and Agencia Estatal de Investigacion.

---

For questions or issues, please open an issue in this repository.
