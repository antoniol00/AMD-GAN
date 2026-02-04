import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

import tensorflow as tf
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REAL_DATA_PATH = '/path/to/data/CIC-IDS2017.csv'
MODELS_DIR_V2 = '/path/to/models/outputs_wgan_multi_v2'
MODELS_DIR_V1 = '/path/to/models/outputs_wgan_multi'
OUTPUT_DIR = '/path/to/output/results_oversampling'
RANDOM_STATE = 42
LATENT_DIM = 100

EXPERIMENT_CLASSES = ['BENIGN', 'Brute Force', 'DDoS', 'DoS', 'Port Scan']

CLASS_TO_FOLDER = {
    'BENIGN': 'benign',
    'Bot': 'bot',
    'Brute Force': 'brute_force',
    'DDoS': 'ddos',
    'DoS': 'dos',
    'Port Scan': 'port_scan',
    'Web Attack': 'web_attack',
}

FEATURES_BASE = [
    'Source Port', 'Destination Port', 'Protocol',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Max Packet Length',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'PSH Flag Count'
]

FEATURE_NAMES = FEATURES_BASE + [
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4'
]

LOG_COLUMNS = [
    'Total Fwd Packets', 'Total Backward Packets', 
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 
    'Packet Length Std', 'Max Packet Length'
]


def print_header():
    print("=" * 100)
    print("  EXPERIMENT: OVERSAMPLING TECHNIQUES COMPARISON")
    print("  GAN (WGAN-GP) vs SMOTE vs ADASYN vs BorderlineSMOTE vs SMOTE-ENN")
    print("=" * 100)


def load_real_data():
    print("\n[1] LOADING REAL DATA")
    print("-" * 50)
    
    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  File: {REAL_DATA_PATH}")
    print(f"  Size: {file_size / (1024 * 1024):.2f} MB")
    
    start_time = datetime.now()
    df_pl = pl.read_csv(REAL_DATA_PATH, low_memory=False)
    df = df_pl.to_pandas()
    df.columns = df.columns.str.strip()
    print(f"  Loaded in {datetime.now() - start_time}")
    print(f"  Total records: {len(df):,}")
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
    df['Label_Class'] = df[label_col]
    
    df = df[df['Label_Class'].isin(EXPERIMENT_CLASSES)].copy()
    
    features_df = df[FEATURES_BASE].copy()
    
    if 'Source IP' in df.columns:
        octets = df['Source IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = 0
    
    if 'Destination IP' in df.columns:
        octets = df['Destination IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = 0
    
    features_df['Label_Class'] = df['Label_Class'].values
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)
    
    return features_df


def show_distribution(y, label_encoder, title="Distribution"):
    print(f"\n  {title}:")
    counter = Counter(y)
    total = len(y)
    for label_idx in sorted(counter.keys()):
        class_name = label_encoder.inverse_transform([label_idx])[0]
        count = counter[label_idx]
        pct = (count / total) * 100
        print(f"    {class_name:<15}: {count:>10,} ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<15}: {total:>10,}")


def oversample_gan(X_train, y_train, label_encoder, target_count):
    print("\n  [GAN] Applying oversampling with WGAN-GP v2...")
    
    X_resampled = X_train.copy()
    y_resampled = y_train.copy()
    
    for class_idx, class_name in enumerate(label_encoder.classes_):
        current_count = np.sum(y_train == class_idx)
        
        if current_count >= target_count:
            print(f"    {class_name}: {current_count:,} >= {target_count:,} (no change)")
            continue
        
        n_generate = target_count - current_count
        print(f"    {class_name}: {current_count:,} -> {target_count:,} (generating {n_generate:,})")
        
        folder = CLASS_TO_FOLDER.get(class_name)
        if not folder:
            print(f"      [WARN] No GAN model for {class_name}")
            continue
        
        generator_path = os.path.join(MODELS_DIR_V2, folder, f'generator_{folder}.h5')
        scaler_path = os.path.join(MODELS_DIR_V2, folder, 'scaler.pkl')
        
        if not os.path.exists(generator_path):
            generator_path = os.path.join(MODELS_DIR_V1, folder, f'generator_{folder}.h5')
            scaler_path = os.path.join(MODELS_DIR_V1, folder, 'scaler.pkl')
        
        if not os.path.exists(generator_path):
            print(f"      [WARN] Generator not found for {class_name}")
            continue
        
        generator = load_model(generator_path, compile=False)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print(f"      [WARN] Scaler not found for {class_name}, using original data")
            continue
        
        noise = np.random.normal(0, 1, (n_generate, LATENT_DIM))
        X_synthetic_scaled = generator.predict(noise, verbose=0)
        
        X_synthetic = scaler.inverse_transform(X_synthetic_scaled)
        
        for i, col in enumerate(FEATURE_NAMES):
            if col in LOG_COLUMNS:
                X_synthetic[:, i] = np.expm1(X_synthetic[:, i])
        
        X_synthetic = np.clip(X_synthetic, 0, None)
        
        X_resampled = np.vstack([X_resampled, X_synthetic])
        y_resampled = np.concatenate([y_resampled, np.full(n_generate, class_idx)])
        
        del generator
        tf.keras.backend.clear_session()
    
    return X_resampled, y_resampled


def oversample_smote(X_train, y_train, label_encoder, target_count):
    print("\n  [SMOTE] Applying SMOTE...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No oversampling required")
        return X_train, y_train
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    for class_idx, class_name in enumerate(label_encoder.classes_):
        old_count = counter[class_idx]
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class_name}: {old_count:,} -> {new_count:,}")
    
    return X_resampled, y_resampled


def oversample_adasyn(X_train, y_train, label_encoder, target_count):
    print("\n  [ADASYN] Applying ADASYN...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No oversampling required")
        return X_train, y_train
    
    try:
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            n_neighbors=5
        )
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        
        for class_idx, class_name in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class_name}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] ADASYN failed: {e}")
        print(f"    Using SMOTE as fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_borderline(X_train, y_train, label_encoder, target_count):
    print("\n  [BorderlineSMOTE] Applying BorderlineSMOTE...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No oversampling required")
        return X_train, y_train
    
    try:
        borderline = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            k_neighbors=5
        )
        X_resampled, y_resampled = borderline.fit_resample(X_train, y_train)
        
        for class_idx, class_name in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class_name}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] BorderlineSMOTE failed: {e}")
        print(f"    Using SMOTE as fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_smoteenn(X_train, y_train, label_encoder, target_count):
    print("\n  [SMOTE-ENN] Applying SMOTE + ENN cleaning...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No oversampling required")
        return X_train, y_train
    
    try:
        smoteenn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        )
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        
        print(f"    Samples before: {len(y_train):,}")
        print(f"    Samples after: {len(y_resampled):,}")
        
        for class_idx, class_name in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            print(f"    {class_name}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] SMOTE-ENN failed: {e}")
        print(f"    Using SMOTE as fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_random(X_train, y_train, label_encoder, target_count):
    print("\n  [RandomOverSampler] Applying random oversampling...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No oversampling required")
        return X_train, y_train
    
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE
    )
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    for class_idx, class_name in enumerate(label_encoder.classes_):
        old_count = counter[class_idx]
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class_name}: {old_count:,} -> {new_count:,}")
    
    return X_resampled, y_resampled


def create_model(model_type='logistic'):
    if model_type == 'logistic':
        return LogisticRegression(
            max_iter=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='multinomial'
        )
    elif model_type == 'tree':
        return DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        return LGBMClassifier(
            n_estimators=200,
            max_depth=15,
            learning_rate=0.1,
            num_leaves=63,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            class_weight='balanced'
        )

MODEL_TYPE = 'logistic'


def train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder, 
                       technique_name, output_dir):
    print(f"\n  Training model with {technique_name}...")
    print(f"    Train: {len(X_train):,} samples")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
    
    model = create_model(MODEL_TYPE)
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = datetime.now() - start_time
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Macro: {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")
    print(f"    Training time: {train_time}")
    
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{technique_name}_classification_report.csv'))
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=label_encoder.classes_,
                         columns=label_encoder.classes_)
    cm_df.to_csv(os.path.join(output_dir, f'{technique_name}_confusion_matrix.csv'))
    
    return {
        'Technique': technique_name,
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
        'Accuracy': accuracy,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'Train_Time': str(train_time),
        'Report': report,
        'Confusion_Matrix': cm
    }


def generate_summary(results, label_encoder, output_dir):
    summary = []
    summary.append("=" * 120)
    summary.append(" " * 25 + "OVERSAMPLING TECHNIQUES COMPARISON")
    summary.append(" " * 30 + "GAN vs Traditional Methods")
    summary.append(" " * 35 + f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 120)
    
    results_sorted = sorted(results, key=lambda x: x['F1_Macro'], reverse=True)
    
    summary.append("\n" + "=" * 120)
    summary.append("1. TECHNIQUES RANKING (sorted by F1-Macro)")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 120)
    header = f"{'Pos':<5} {'Technique':<20} | {'Train':>12} | {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weight':>10} | {'Prec':>8} {'Recall':>8}"
    summary.append(header)
    summary.append("-" * 120)
    
    best = results_sorted[0]
    for i, r in enumerate(results_sorted, 1):
        delta = r['F1_Macro'] - best['F1_Macro']
        delta_str = f"({delta:+.4f})" if i > 1 else "(best)"
        line = f"{i:<5} {r['Technique']:<20} | {r['Train_Samples']:>12,} | {r['Accuracy']:>10.4f} {r['F1_Macro']:>10.4f} {r['F1_Weighted']:>10.4f} | {r['Precision_Macro']:>8.4f} {r['Recall_Macro']:>8.4f} {delta_str}"
        summary.append(line)
    
    summary.append("\n\n" + "=" * 120)
    summary.append("2. GAN vs OTHER METHODS COMPARISON")
    summary.append("=" * 120)
    
    gan_result = next((r for r in results if r['Technique'] == 'GAN_v2'), None)
    baseline_result = next((r for r in results if r['Technique'] == 'Original'), None)
    smote_result = next((r for r in results if r['Technique'] == 'SMOTE'), None)
    
    if gan_result and baseline_result:
        improvement_vs_baseline = gan_result['F1_Macro'] - baseline_result['F1_Macro']
        summary.append(f"\n  GAN vs Original (no oversampling):")
        summary.append(f"    - Delta F1-Macro: {improvement_vs_baseline:+.4f}")
        summary.append(f"    - Delta Accuracy: {gan_result['Accuracy'] - baseline_result['Accuracy']:+.4f}")
    
    if gan_result and smote_result:
        improvement_vs_smote = gan_result['F1_Macro'] - smote_result['F1_Macro']
        summary.append(f"\n  GAN vs SMOTE:")
        summary.append(f"    - Delta F1-Macro: {improvement_vs_smote:+.4f}")
        summary.append(f"    - Delta Accuracy: {gan_result['Accuracy'] - smote_result['Accuracy']:+.4f}")
    
    summary.append("\n\n" + "=" * 120)
    summary.append(f"3. PER-CLASS ANALYSIS - BEST TECHNIQUE: {best['Technique']}")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 90)
    summary.append(f"{'Class':<15} | {'Precision':>10} {'Recall':>10} {'F1-Score':>10} | {'Support':>10}")
    summary.append("-" * 90)
    
    report = best['Report']
    for class_name in label_encoder.classes_:
        if class_name in report:
            r = report[class_name]
            summary.append(f"{class_name:<15} | {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} | {r['support']:>10.0f}")
    
    summary.append("\n\n" + "=" * 120)
    summary.append("4. F1-SCORE PER CLASS FOR EACH TECHNIQUE")
    summary.append("=" * 120)
    
    technique_names = [r['Technique'][:12] for r in results]
    header_line = f"{'Class':<15} | " + " | ".join([f"{t:>12}" for t in technique_names])
    summary.append("\n" + "-" * len(header_line))
    summary.append(header_line)
    summary.append("-" * len(header_line))
    
    for class_name in label_encoder.classes_:
        values = []
        for r in results:
            if class_name in r['Report']:
                values.append(f"{r['Report'][class_name]['f1-score']:>12.4f}")
            else:
                values.append(f"{'N/A':>12}")
        line = f"{class_name:<15} | " + " | ".join(values)
        summary.append(line)
    
    summary.append("\n\n" + "=" * 120)
    summary.append("5. CONCLUSIONS")
    summary.append("=" * 120)
    
    summary.append("\n+" + "-" * 111 + "+")
    summary.append(f"|  BEST TECHNIQUE: {best['Technique']:<20} F1-Macro = {best['F1_Macro']:.4f}" + " " * 50 + "|")
    summary.append("+" + "-" * 111 + "+")
    
    if gan_result:
        gan_pos = next(i for i, r in enumerate(results_sorted, 1) if r['Technique'] == 'GAN_v2')
        if gan_pos == 1:
            summary.append(f"|  [OK] GAN outperforms all traditional oversampling techniques" + " " * 48 + "|")
        elif gan_pos <= 3:
            summary.append(f"|  [--] GAN in position {gan_pos} - Competitive with traditional methods" + " " * 43 + "|")
        else:
            summary.append(f"|  [XX] GAN in position {gan_pos} - Traditional methods work better in this case" + " " * 35 + "|")
    
    summary.append("+" + "-" * 111 + "+")
    
    summary_text = "\n".join(summary)
    
    with open(os.path.join(output_dir, 'COMPARISON_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    
    comparison = [{
        'Technique': r['Technique'],
        'Train_Samples': r['Train_Samples'],
        'Accuracy': r['Accuracy'],
        'F1_Macro': r['F1_Macro'],
        'F1_Weighted': r['F1_Weighted'],
        'Precision_Macro': r['Precision_Macro'],
        'Recall_Macro': r['Recall_Macro'],
        'Train_Time': r['Train_Time']
    } for r in results]
    
    pd.DataFrame(comparison).to_csv(
        os.path.join(output_dir, 'techniques_comparison.csv'), 
        index=False
    )
    
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description='Oversampling techniques comparison: GAN vs traditional methods'
    )
    parser.add_argument('--max-samples', type=int, default=100000,
                       help='Max samples per class for balancing (default: 100000)')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set proportion (default: 0.3)')
    parser.add_argument('--skip-gan', action='store_true',
                       help='Skip GAN oversampling')
    parser.add_argument('--output', type=str, default=None,
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    print_header()
    
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f'comparison_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Results will be saved to: {output_dir}")
    
    df = load_real_data()
    df_features = prepare_features(df)
    
    print(f"\n[2] CLASS DISTRIBUTION IN FILTERED DATASET")
    print("-" * 50)
    print(f"  Classes: {EXPERIMENT_CLASSES}")
    print(f"  Total samples: {len(df_features):,}")
    print("\n  Distribution:")
    for class_name in EXPERIMENT_CLASSES:
        count = len(df_features[df_features['Label_Class'] == class_name])
        pct = (count / len(df_features)) * 100
        print(f"    {class_name:<15}: {count:>10,} ({pct:>5.1f}%)")
    
    feature_cols = [c for c in df_features.columns if c != 'Label_Class']
    X = df_features[feature_cols].values
    
    label_encoder = LabelEncoder()
    label_encoder.fit(EXPERIMENT_CLASSES)
    y = label_encoder.transform(df_features['Label_Class'])
    
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Encoded classes: {list(label_encoder.classes_)}")
    
    print(f"\n[3] SPLITTING TRAIN/TEST")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    show_distribution(y_train, label_encoder, "Train Distribution (original)")
    show_distribution(y_test, label_encoder, "Test Distribution")
    
    counter = Counter(y_train)
    max_class_count = max(counter.values())
    target_count = min(max_class_count, args.max_samples)
    
    print(f"\n  Balancing target: {target_count:,} samples per class")
    
    print(f"\n[4] APPLYING OVERSAMPLING TECHNIQUES")
    print("=" * 70)
    
    techniques = {}
    
    print("\n" + "-" * 70)
    print("  [Original] No oversampling (baseline)")
    techniques['Original'] = (X_train.copy(), y_train.copy())
    show_distribution(y_train, label_encoder, "Distribution")
    
    if not args.skip_gan:
        print("\n" + "-" * 70)
        X_gan, y_gan = oversample_gan(X_train, y_train, label_encoder, target_count)
        techniques['GAN_v2'] = (X_gan, y_gan)
        show_distribution(y_gan, label_encoder, "GAN Distribution")
    
    print("\n" + "-" * 70)
    X_smote, y_smote = oversample_smote(X_train, y_train, label_encoder, target_count)
    techniques['SMOTE'] = (X_smote, y_smote)
    show_distribution(y_smote, label_encoder, "SMOTE Distribution")
    
    print("\n" + "-" * 70)
    X_adasyn, y_adasyn = oversample_adasyn(X_train, y_train, label_encoder, target_count)
    techniques['ADASYN'] = (X_adasyn, y_adasyn)
    show_distribution(y_adasyn, label_encoder, "ADASYN Distribution")
    
    print("\n" + "-" * 70)
    X_borderline, y_borderline = oversample_borderline(X_train, y_train, label_encoder, target_count)
    techniques['BorderlineSMOTE'] = (X_borderline, y_borderline)
    show_distribution(y_borderline, label_encoder, "BorderlineSMOTE Distribution")
    
    print("\n" + "-" * 70)
    X_smoteenn, y_smoteenn = oversample_smoteenn(X_train, y_train, label_encoder, target_count)
    techniques['SMOTE_ENN'] = (X_smoteenn, y_smoteenn)
    show_distribution(y_smoteenn, label_encoder, "SMOTE-ENN Distribution")
    
    print("\n" + "-" * 70)
    X_random, y_random = oversample_random(X_train, y_train, label_encoder, target_count)
    techniques['RandomOverSampler'] = (X_random, y_random)
    show_distribution(y_random, label_encoder, "Random Distribution")
    
    print(f"\n\n[5] TRAINING AND EVALUATION")
    print("=" * 70)
    model_desc = {
        'logistic': 'Logistic Regression (multinomial, balanced)',
        'tree': 'Decision Tree (max_depth=10, balanced)',
        'lgbm': 'LightGBM (200 estimators, max_depth=15)'
    }
    print(f"  Model: {model_desc.get(MODEL_TYPE, MODEL_TYPE)}")
    print(f"  Test set: {len(X_test):,} samples (same for all techniques)")
    
    results = []
    
    for name, (X_t, y_t) in techniques.items():
        print(f"\n" + "-" * 70)
        result = train_and_evaluate(
            X_t, y_t, X_test, y_test,
            label_encoder, name, output_dir
        )
        results.append(result)
    
    print(f"\n\n[6] GENERATING COMPARISON SUMMARY")
    print("=" * 70)
    
    generate_summary(results, label_encoder, output_dir)
    
    print(f"\n\n{'='*100}")
    print(f"  EXPERIMENT COMPLETED")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
