import os
import sys
import argparse
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, BaggingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

REAL_DATA_PATH = '/path/to/data/CIC-IDS2017.csv'
SYNTHETIC_DATASETS_DIR = '/path/to/output/generated_datasets'
OUTPUT_BASE_DIR = '/path/to/output/results_tstr'
RANDOM_STATE = 42


def list_available_datasets():
    if not os.path.exists(SYNTHETIC_DATASETS_DIR):
        print(f"[ERROR] Directory does not exist: {SYNTHETIC_DATASETS_DIR}")
        return []
    
    datasets = []
    for f in os.listdir(SYNTHETIC_DATASETS_DIR):
        if f.endswith('.csv') and not f.endswith('_config.json'):
            csv_path = os.path.join(SYNTHETIC_DATASETS_DIR, f)
            config_path = csv_path.replace('.csv', '_config.json')
            
            try:
                df_sample = pd.read_csv(csv_path, nrows=5)
                n_rows = sum(1 for _ in open(csv_path)) - 1
                
                config = {}
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as cf:
                        config = json.load(cf)
                
                datasets.append({
                    'name': f.replace('.csv', ''),
                    'file': f,
                    'path': csv_path,
                    'rows': n_rows,
                    'config': config,
                    'classes': config.get('samples_per_class', {})
                })
            except Exception as e:
                print(f"  [WARN] Error reading {f}: {e}")
    
    return datasets


def show_available_datasets(datasets):
    print("\n" + "=" * 80)
    print("AVAILABLE SYNTHETIC DATASETS")
    print("=" * 80)
    
    if not datasets:
        print("  No datasets available in:", SYNTHETIC_DATASETS_DIR)
        print("  Generate one with: python generate_synthetic_dataset.py --interactive")
        return
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n  [{i}] {ds['name']}")
        print(f"      File: {ds['file']}")
        print(f"      Total samples: {ds['rows']:,}")
        if ds['classes']:
            print(f"      Classes: {', '.join(ds['classes'].keys())}")
            for cls, n in ds['classes'].items():
                print(f"        - {cls}: {n:,}")
    
    print("\n" + "=" * 80)


def select_dataset_interactive(datasets):
    show_available_datasets(datasets)
    
    if not datasets:
        return None
    
    while True:
        try:
            inp = input("\nSelect a dataset (number or name): ").strip()
            
            if inp.isdigit():
                idx = int(inp) - 1
                if 0 <= idx < len(datasets):
                    return datasets[idx]
                print(f"  [!] Number must be between 1 and {len(datasets)}")
                continue
            
            for ds in datasets:
                if ds['name'].lower() == inp.lower() or ds['file'].lower() == inp.lower():
                    return ds
            
            print(f"  [!] Dataset '{inp}' not found")
            
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return None


def load_synthetic_dataset(dataset_info) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("LOADING SYNTHETIC DATASET")
    print("=" * 70)
    
    print(f"  File: {dataset_info['file']}")
    
    df = pd.read_csv(dataset_info['path'])
    
    print(f"  Total samples: {len(df):,}")
    
    if 'Attack Type' in df.columns:
        label_col = 'Attack Type'
    elif 'Label_Class' in df.columns:
        label_col = 'Label_Class'
    else:
        raise ValueError("Class column not found (Attack Type or Label_Class)")
    
    df['Label_Class'] = df[label_col]
    
    print(f"\n  Class distribution:")
    for cls, count in df['Label_Class'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    return df


def load_real_data() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("LOADING REAL DATA")
    print("=" * 70)
    
    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
    
    start_time = datetime.now()
    df_pl = pl.read_csv(REAL_DATA_PATH, low_memory=False)
    df = df_pl.to_pandas()
    print(f"  Data loaded in {datetime.now() - start_time}")
    
    return df


def prepare_real_features(df: pd.DataFrame) -> pd.DataFrame:
    FEATURES_BASE = [
        'Source IP', 'Destination IP',
        'Source Port', 'Destination Port', 'Protocol',
        'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Packet Length Std', 'Max Packet Length',
        'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count'
    ]
    
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
    df['Label_Class'] = df[label_col]
    
    available_features = [f for f in FEATURES_BASE if f in df.columns]
    df = df[available_features + ['Label_Class']].copy()
    
    if 'Source IP' in df.columns:
        octets = df['Source IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            df[f'Src_IP_{i+1}'] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
        df.drop(columns=['Source IP'], inplace=True)
    
    if 'Destination IP' in df.columns:
        octets = df['Destination IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            df[f'Dst_IP_{i+1}'] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
        df.drop(columns=['Destination IP'], inplace=True)
    
    return df


def filter_classes(df: pd.DataFrame, target_classes: list) -> pd.DataFrame:
    df_filtered = df[df['Label_Class'].isin(target_classes)].copy()
    print(f"\n  Rows after filtering target classes: {len(df_filtered):,} of {len(df):,}")
    return df_filtered


def prepare_data_for_training(df_synth: pd.DataFrame, df_real: pd.DataFrame, label_encoder: LabelEncoder):
    cols_synth = set(df_synth.columns) - {'Label', 'Label_Class', 'Attack Type'}
    cols_real = set(df_real.columns) - {'Label', 'Label_Class', 'Attack Type'}
    feature_cols = sorted(list(cols_synth.intersection(cols_real)))
    
    print(f"\n  Common features for training: {len(feature_cols)}")
    
    X_synth = df_synth[feature_cols].values
    y_synth = label_encoder.transform(df_synth['Label_Class'])
    
    X_real = df_real[feature_cols].values
    y_real = label_encoder.transform(df_real['Label_Class'])
    
    X_synth = np.nan_to_num(X_synth, nan=0, posinf=0, neginf=0)
    X_real = np.nan_to_num(X_real, nan=0, posinf=0, neginf=0)
    
    return X_synth, y_synth, X_real, y_real, feature_cols


def train_and_evaluate_model(model, name, X_train, y_train, X_test, y_test, 
                               label_encoder, output_dir, prefix=""):
    print(f"\n  Training {name}...", end=" ")
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = datetime.now() - start_time
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Acc: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}, F1-Weighted: {f1_weighted:.4f}")
    
    report = classification_report(y_test, y_pred, 
                                   target_names=label_encoder.classes_,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{prefix}{name}_classification_report.csv'))
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=label_encoder.classes_, 
                        columns=label_encoder.classes_)
    cm_df.to_csv(os.path.join(output_dir, f'{prefix}{name}_confusion_matrix.csv'))
    
    return {
        'Model': name,
        'Accuracy': accuracy,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Train_Time': str(train_time),
        'Train_Samples': len(y_train),
        'Test_Samples': len(y_test),
        'Report': report,
        'Confusion_Matrix': cm
    }


def run_experiment(X_train, y_train, X_test, y_test, label_encoder, output_dir, prefix, description):
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print('='*70)
    
    models = {
        'Dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'LogisticReg': LogisticRegression(multi_class='multinomial', max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced_subsample'),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, use_label_encoder=False),
        'Voting': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
                ('lgbm', LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
            ],
            voting='soft', n_jobs=-1
        ),
    }
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    for name, model in models.items():
        result = train_and_evaluate_model(
            model, name, X_train_scaled, y_train, X_test_scaled, y_test,
            label_encoder, output_dir, prefix
        )
        result['Experiment'] = description
        results.append(result)
    
    return results


def generate_global_summary(results_tstr_multi, results_trtr_multi,
                           results_tstr_binary, results_trtr_binary,
                           label_encoder_multi, label_encoder_binary,
                           output_dir, dataset_name):
    
    summary = []
    summary.append("=" * 120)
    summary.append(" " * 30 + "GLOBAL SUMMARY: ALL EXPERIMENTS")
    summary.append(" " * 20 + "TSTR vs TRTR - Multiclass and Binary Classification")
    summary.append(f" " * 30 + f"Dataset: {dataset_name}")
    summary.append(f" " * 45 + f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 120)
    
    summary.append("\n" + "=" * 120)
    summary.append("1. EXECUTIVE SUMMARY")
    summary.append("=" * 120)
    
    best_tstr_multi = max(results_tstr_multi, key=lambda x: x['F1_Macro'])
    best_trtr_multi = max(results_trtr_multi, key=lambda x: x['F1_Macro'])
    best_tstr_binary = max(results_tstr_binary, key=lambda x: x['F1_Macro'])
    best_trtr_binary = max(results_trtr_binary, key=lambda x: x['F1_Macro'])
    
    gap_multi = best_tstr_multi['F1_Macro'] - best_trtr_multi['F1_Macro']
    gap_binary = best_tstr_binary['F1_Macro'] - best_trtr_binary['F1_Macro']
    
    summary.append(f"\nClasses used: {', '.join(label_encoder_multi.classes_)}")
    
    summary.append("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    summary.append("│                                    BEST RESULTS PER EXPERIMENT                                                  │")
    summary.append("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    summary.append(f"│  MULTICLASS ({len(label_encoder_multi.classes_)} classes):                                                                                           │")
    summary.append(f"│    • TSTR: {best_tstr_multi['Model']:<15} F1-Macro = {best_tstr_multi['F1_Macro']:.4f}  Accuracy = {best_tstr_multi['Accuracy']:.4f}                                        │")
    summary.append(f"│    • TRTR: {best_trtr_multi['Model']:<15} F1-Macro = {best_trtr_multi['F1_Macro']:.4f}  Accuracy = {best_trtr_multi['Accuracy']:.4f}  (baseline)                            │")
    summary.append(f"│    • Gap TSTR-TRTR: {gap_multi:+.4f}                                                                                │")
    summary.append(f"│                                                                                                                 │")
    summary.append(f"│  BINARY:                                                                                                        │")
    summary.append(f"│    • TSTR: {best_tstr_binary['Model']:<15} F1-Macro = {best_tstr_binary['F1_Macro']:.4f}  Accuracy = {best_tstr_binary['Accuracy']:.4f}                                        │")
    summary.append(f"│    • TRTR: {best_trtr_binary['Model']:<15} F1-Macro = {best_trtr_binary['F1_Macro']:.4f}  Accuracy = {best_trtr_binary['Accuracy']:.4f}  (baseline)                            │")
    summary.append(f"│    • Gap TSTR-TRTR: {gap_binary:+.4f}                                                                                │")
    summary.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    summary.append("\n\n" + "=" * 120)
    summary.append("2. DETAILED COMPARISON - MULTICLASS CLASSIFICATION")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 120)
    header = f"{'Model':<20} │ {'TSTR Acc':>10} {'TSTR F1-M':>10} {'TSTR F1-W':>10} │ {'TRTR Acc':>10} {'TRTR F1-M':>10} {'TRTR F1-W':>10} │ {'Δ F1-M':>8} {'Δ Acc':>8}"
    summary.append(header)
    summary.append("-" * 120)
    
    for r_tstr in results_tstr_multi:
        model = r_tstr['Model']
        r_trtr = next((r for r in results_trtr_multi if r['Model'] == model), None)
        if r_trtr:
            delta_f1 = r_tstr['F1_Macro'] - r_trtr['F1_Macro']
            delta_acc = r_tstr['Accuracy'] - r_trtr['Accuracy']
            line = f"{model:<20} │ {r_tstr['Accuracy']:>10.4f} {r_tstr['F1_Macro']:>10.4f} {r_tstr['F1_Weighted']:>10.4f} │ {r_trtr['Accuracy']:>10.4f} {r_trtr['F1_Macro']:>10.4f} {r_trtr['F1_Weighted']:>10.4f} │ {delta_f1:>+8.4f} {delta_acc:>+8.4f}"
            summary.append(line)
    
    summary.append("\n\n" + "=" * 120)
    summary.append("3. DETAILED COMPARISON - BINARY CLASSIFICATION (BENIGN vs ATTACK)")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 120)
    summary.append(header)
    summary.append("-" * 120)
    
    for r_tstr in results_tstr_binary:
        model = r_tstr['Model']
        r_trtr = next((r for r in results_trtr_binary if r['Model'] == model), None)
        if r_trtr:
            delta_f1 = r_tstr['F1_Macro'] - r_trtr['F1_Macro']
            delta_acc = r_tstr['Accuracy'] - r_trtr['Accuracy']
            line = f"{model:<20} │ {r_tstr['Accuracy']:>10.4f} {r_tstr['F1_Macro']:>10.4f} {r_tstr['F1_Weighted']:>10.4f} │ {r_trtr['Accuracy']:>10.4f} {r_trtr['F1_Macro']:>10.4f} {r_trtr['F1_Weighted']:>10.4f} │ {delta_f1:>+8.4f} {delta_acc:>+8.4f}"
            summary.append(line)
    
    summary.append("\n\n" + "=" * 120)
    summary.append("4. PER-CLASS ANALYSIS - BEST TSTR MULTICLASS MODEL")
    summary.append("=" * 120)
    
    report_tstr = best_tstr_multi['Report']
    report_trtr = best_trtr_multi['Report']
    
    summary.append(f"\nModel: {best_tstr_multi['Model']}")
    summary.append("\n" + "-" * 90)
    summary.append(f"{'Class':<15} │ {'TSTR Prec':>10} {'TSTR Rec':>10} {'TSTR F1':>10} │ {'TRTR F1':>10} │ {'Δ F1':>8}")
    summary.append("-" * 90)
    
    for cls in label_encoder_multi.classes_:
        if cls in report_tstr and cls in report_trtr:
            tstr_r = report_tstr[cls]
            trtr_r = report_trtr[cls]
            delta = tstr_r['f1-score'] - trtr_r['f1-score']
            summary.append(f"{cls:<15} │ {tstr_r['precision']:>10.4f} {tstr_r['recall']:>10.4f} {tstr_r['f1-score']:>10.4f} │ {trtr_r['f1-score']:>10.4f} │ {delta:>+8.4f}")
    
    summary.append("\n\n" + "=" * 120)
    summary.append("5. CONFUSION MATRIX - TSTR MULTICLASS (Best model: " + best_tstr_multi['Model'] + ")")
    summary.append("=" * 120)
    
    cm = best_tstr_multi['Confusion_Matrix']
    
    short_classes = [c[:8] for c in label_encoder_multi.classes_]
    header_cm = "Pred→    " + "".join([f"{c:>10}" for c in short_classes])
    summary.append("\n" + header_cm)
    summary.append("Real↓    " + "-" * (10 * len(short_classes)))
    
    for i, cls in enumerate(label_encoder_multi.classes_):
        row = f"{cls[:8]:<9}" + "".join([f"{cm[i, j]:>10}" for j in range(len(label_encoder_multi.classes_))])
        summary.append(row)
    
    summary.append("\n\n" + "=" * 120)
    summary.append("6. INSIGHTS AND CONCLUSIONS")
    summary.append("=" * 120)
    
    summary.append("\n┌─ SYNTHETIC DATA QUALITY ───────────────────────────────────────────────────────────────────────────────────────┐")
    
    if best_tstr_binary['F1_Macro'] > 0.95:
        summary.append(f"│  ✓ EXCELLENT: TSTR Binary F1-Macro = {best_tstr_binary['F1_Macro']:.4f} (>0.95)                                                     │")
    elif best_tstr_binary['F1_Macro'] > 0.90:
        summary.append(f"│  ✓ VERY GOOD: TSTR Binary F1-Macro = {best_tstr_binary['F1_Macro']:.4f} (>0.90)                                                     │")
    else:
        summary.append(f"│  ○ ACCEPTABLE: TSTR Binary F1-Macro = {best_tstr_binary['F1_Macro']:.4f}                                                            │")
    
    if best_tstr_multi['F1_Macro'] > 0.85:
        summary.append(f"│  ✓ SUCCESS: TSTR Multiclass F1-Macro = {best_tstr_multi['F1_Macro']:.4f} (>0.85)                                                    │")
    else:
        summary.append(f"│  ○ TSTR Multiclass F1-Macro = {best_tstr_multi['F1_Macro']:.4f} (<0.85) - Room for improvement                                      │")
    summary.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    summary.append("\n┌─ PROBLEMATIC CLASSES IN TSTR MULTICLASS ───────────────────────────────────────────────────────────────────────┐")
    
    classes_f1 = [(c, report_tstr[c]['f1-score']) for c in label_encoder_multi.classes_ if c in report_tstr]
    classes_f1_sorted = sorted(classes_f1, key=lambda x: x[1])
    
    for cls, f1 in classes_f1_sorted[:3]:
        if f1 < 0.5:
            summary.append(f"│  ✗ {cls}: F1 = {f1:.4f} - CRITICAL                                                                           │")
        elif f1 < 0.8:
            summary.append(f"│  ○ {cls}: F1 = {f1:.4f} - NEEDS IMPROVEMENT                                                                   │")
        else:
            summary.append(f"│  ✓ {cls}: F1 = {f1:.4f} - ACCEPTABLE                                                                          │")
    summary.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    summary.append("\n\n" + "=" * 120)
    summary.append("7. FINAL NUMERICAL SUMMARY")
    summary.append("=" * 120)
    
    summary.append("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    summary.append("│                                         KEY METRICS                                                             │")
    summary.append("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    summary.append(f"│  BINARY:                                                                                                        │")
    summary.append(f"│    • TSTR F1-Macro: {best_tstr_binary['F1_Macro']:.4f}    TRTR F1-Macro: {best_trtr_binary['F1_Macro']:.4f}    Gap: {gap_binary:+.4f}                                    │")
    summary.append(f"│    • TSTR Accuracy: {best_tstr_binary['Accuracy']:.4f}    TRTR Accuracy: {best_trtr_binary['Accuracy']:.4f}                                                  │")
    summary.append(f"│                                                                                                                 │")
    summary.append(f"│  MULTICLASS:                                                                                                    │")
    summary.append(f"│    • TSTR F1-Macro: {best_tstr_multi['F1_Macro']:.4f}    TRTR F1-Macro: {best_trtr_multi['F1_Macro']:.4f}    Gap: {gap_multi:+.4f}                                    │")
    summary.append(f"│    • TSTR Accuracy: {best_tstr_multi['Accuracy']:.4f}    TRTR Accuracy: {best_trtr_multi['Accuracy']:.4f}                                                  │")
    summary.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    summary_text = "\n".join(summary)
    
    with open(os.path.join(output_dir, 'GLOBAL_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    
    global_comparison = []
    
    for exp_name, results in [
        ('TSTR_Multi', results_tstr_multi),
        ('TRTR_Multi', results_trtr_multi),
        ('TSTR_Binary', results_tstr_binary),
        ('TRTR_Binary', results_trtr_binary)
    ]:
        for r in results:
            global_comparison.append({
                'Experiment': exp_name,
                'Model': r['Model'],
                'Accuracy': r['Accuracy'],
                'F1_Macro': r['F1_Macro'],
                'F1_Weighted': r['F1_Weighted']
            })
    
    pd.DataFrame(global_comparison).to_csv(os.path.join(output_dir, 'global_comparison.csv'), index=False)
    
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description='TSTR vs TRTR evaluation with synthetic datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', '-d', type=str, help='Name of synthetic dataset to use')
    parser.add_argument('--list', '-l', action='store_true', help='List available datasets')
    parser.add_argument('--max-train', type=int, default=100000, help='Max training samples')
    parser.add_argument('--max-test', type=int, default=100000, help='Max test samples')
    
    args = parser.parse_args()
    
    datasets = list_available_datasets()
    
    if args.list:
        show_available_datasets(datasets)
        return
    
    print("=" * 100)
    print("COMPLETE EXPERIMENT: TSTR vs TRTR - MULTICLASS AND BINARY")
    print("=" * 100)
    
    if args.dataset:
        dataset_info = None
        for ds in datasets:
            if ds['name'].lower() == args.dataset.lower() or ds['file'].lower() == args.dataset.lower():
                dataset_info = ds
                break
        
        if not dataset_info:
            print(f"\n[ERROR] Dataset '{args.dataset}' not found.")
            show_available_datasets(datasets)
            return
    else:
        dataset_info = select_dataset_interactive(datasets)
        
        if not dataset_info:
            return
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_info['name'])
    os.makedirs(output_dir, exist_ok=True)
    
    df_synth = load_synthetic_dataset(dataset_info)
    
    synthetic_classes = sorted(df_synth['Label_Class'].unique().tolist())
    print(f"\n  Classes detected in synthetic dataset: {synthetic_classes}")
    
    df_real_raw = load_real_data()
    
    df_real = prepare_real_features(df_real_raw)
    
    df_real = filter_classes(df_real, synthetic_classes)
    
    print(f"\n  Real class distribution (filtered):")
    print(df_real['Label_Class'].value_counts())
    
    label_encoder_multi = LabelEncoder()
    label_encoder_multi.fit(synthetic_classes)
    print(f"\n  Multiclass encoder classes: {label_encoder_multi.classes_}")
    
    label_encoder_binary = LabelEncoder()
    label_encoder_binary.fit(['ATTACK', 'BENIGN'])
    print(f"  Binary encoder classes: {label_encoder_binary.classes_}")
    
    X_synth, y_synth_multi, X_real, y_real_multi, feature_cols = prepare_data_for_training(
        df_synth, df_real, label_encoder_multi
    )
    
    benign_idx = np.where(label_encoder_multi.classes_ == 'BENIGN')[0]
    if len(benign_idx) == 0:
        print("\n[WARNING] 'BENIGN' class not found. Binary classification will use first class as 'non-attack'.")
        benign_idx = 0
    else:
        benign_idx = benign_idx[0]
    
    y_synth_binary = np.where(y_synth_multi == benign_idx, 1, 0)
    y_real_binary = np.where(y_real_multi == benign_idx, 1, 0)
    
    np.random.seed(RANDOM_STATE)
    
    X_real_train, X_real_test, y_real_train_multi, y_real_test_multi = train_test_split(
        X_real, y_real_multi, test_size=0.3, random_state=RANDOM_STATE, stratify=y_real_multi
    )
    
    y_real_train_binary = np.where(y_real_train_multi == benign_idx, 1, 0)
    y_real_test_binary = np.where(y_real_test_multi == benign_idx, 1, 0)
    
    max_train = args.max_train
    max_test = args.max_test
    
    if len(X_real_train) > max_train:
        idx_train = []
        for cls in range(len(label_encoder_multi.classes_)):
            idx_cls = np.where(y_real_train_multi == cls)[0]
            n_sample = min(len(idx_cls), max_train // len(label_encoder_multi.classes_))
            if n_sample > 0:
                idx_train.extend(np.random.choice(idx_cls, n_sample, replace=False))
        idx_train = np.array(idx_train)
        np.random.shuffle(idx_train)
        X_real_train = X_real_train[idx_train]
        y_real_train_multi = y_real_train_multi[idx_train]
        y_real_train_binary = y_real_train_binary[idx_train]
    
    if len(X_real_test) > max_test:
        idx_test = np.random.choice(len(X_real_test), max_test, replace=False)
        X_real_test = X_real_test[idx_test]
        y_real_test_multi = y_real_test_multi[idx_test]
        y_real_test_binary = y_real_test_binary[idx_test]
    
    print(f"\n{'='*100}")
    print("DATA SUMMARY FOR ALL EXPERIMENTS")
    print('='*100)
    print(f"  Synthetic dataset: {dataset_info['name']}")
    print(f"  Synthetic Data (Train TSTR): {len(X_synth):,} samples")
    print(f"  Real Data Train (TRTR): {len(X_real_train):,} samples")
    print(f"  Real Data Test (both): {len(X_real_test):,} samples")
    print(f"  Features: {len(feature_cols)}")
    
    print(f"\n  MULTICLASS distribution in Synthetic Train:")
    for i, cls in enumerate(label_encoder_multi.classes_):
        count = np.sum(y_synth_multi == i)
        print(f"    {cls}: {count:,}")
    
    print(f"\n  BINARY distribution in Synthetic Train:")
    print(f"    BENIGN: {np.sum(y_synth_binary == 1):,}")
    print(f"    ATTACK: {np.sum(y_synth_binary == 0):,}")
    
    print("\n\n" + "#"*100)
    print("#" + " "*35 + "MULTICLASS CLASSIFICATION" + " "*36 + "#")
    print("#"*100)
    
    results_tstr_multi = run_experiment(
        X_synth, y_synth_multi, X_real_test, y_real_test_multi,
        label_encoder_multi, output_dir, "TSTR_MULTI_",
        "EXPERIMENT 1: TSTR MULTICLASS (Train Synthetic, Test Real)"
    )
    
    results_trtr_multi = run_experiment(
        X_real_train, y_real_train_multi, X_real_test, y_real_test_multi,
        label_encoder_multi, output_dir, "TRTR_MULTI_",
        "EXPERIMENT 2: TRTR MULTICLASS (Train Real, Test Real) - BASELINE"
    )
    
    print("\n\n" + "#"*100)
    print("#" + " "*37 + "BINARY CLASSIFICATION" + " "*38 + "#")
    print("#"*100)
    
    results_tstr_binary = run_experiment(
        X_synth, y_synth_binary, X_real_test, y_real_test_binary,
        label_encoder_binary, output_dir, "TSTR_BINARY_",
        "EXPERIMENT 3: TSTR BINARY (Train Synthetic, Test Real)"
    )
    
    results_trtr_binary = run_experiment(
        X_real_train, y_real_train_binary, X_real_test, y_real_test_binary,
        label_encoder_binary, output_dir, "TRTR_BINARY_",
        "EXPERIMENT 4: TRTR BINARY (Train Real, Test Real) - BASELINE"
    )
    
    generate_global_summary(
        results_tstr_multi, results_trtr_multi,
        results_tstr_binary, results_trtr_binary,
        label_encoder_multi, label_encoder_binary,
        output_dir, dataset_info['name']
    )
    
    all_results = []
    
    for r in results_tstr_multi:
        r_copy = {k: v for k, v in r.items() if k not in ['Report', 'Confusion_Matrix']}
        r_copy['Type'] = 'TSTR'
        r_copy['Classification'] = 'Multiclass'
        all_results.append(r_copy)
    
    for r in results_trtr_multi:
        r_copy = {k: v for k, v in r.items() if k not in ['Report', 'Confusion_Matrix']}
        r_copy['Type'] = 'TRTR'
        r_copy['Classification'] = 'Multiclass'
        all_results.append(r_copy)
    
    for r in results_tstr_binary:
        r_copy = {k: v for k, v in r.items() if k not in ['Report', 'Confusion_Matrix']}
        r_copy['Type'] = 'TSTR'
        r_copy['Classification'] = 'Binary'
        all_results.append(r_copy)
    
    for r in results_trtr_binary:
        r_copy = {k: v for k, v in r.items() if k not in ['Report', 'Confusion_Matrix']}
        r_copy['Type'] = 'TRTR'
        r_copy['Classification'] = 'Binary'
        all_results.append(r_copy)
    
    pd.DataFrame(all_results).to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    print(f"\n\n{'='*100}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print('='*100)


if __name__ == "__main__":
    main()
