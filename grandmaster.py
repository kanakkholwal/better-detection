# ==============================================================================
# ECG GRANDMASTER PIPELINE: 5-Paradigm Approach
# Target: >99% Accuracy via Orthogonal Model Ensembling
# ==============================================================================

import ast
import os
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
import wfdb
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, mixed_precision, models, optimizers
from xgboost import XGBClassifier

# 1. OPTIMIZATIONS
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision Enabled (GPU Speedup).")
except:
    pass

CONFIG = {
    'SAMP_RATE': 100,
    'DURATION': 10,
    'OUTPUT_DIR': './grandmaster_results'
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING (THREADED)
# ==============================================================================
def load_and_prep_data():
    print("--- 1. Acquiring Data ---")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    
    # Locate CSV
    csv_path = None
    for root, _, files in os.walk(path):
        if 'ptbxl_database.csv' in files:
            csv_path = os.path.join(root, 'ptbxl_database.csv')
            data_root = root
            break
    
    Y = pd.read_csv(csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Label Parsing
    agg_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def get_label(y_dic):
        tmp = [agg_df.loc[k].diagnostic_class for k in y_dic if k in agg_df.index]
        return list(set(tmp))

    Y['diagnostic'] = Y.scp_codes.apply(get_label)
    
    # Binary Filter: MI vs NORM
    mi_df = Y[Y['diagnostic'].apply(lambda x: 'MI' in x)].copy()
    norm_df = Y[Y['diagnostic'].apply(lambda x: 'NORM' in x)].copy()
    
    mi_df['target'] = 1
    # Balance Dataset
    if len(norm_df) > len(mi_df):
        norm_df = norm_df.sample(n=len(mi_df), random_state=42)
    norm_df['target'] = 0
    
    df = pd.concat([mi_df, norm_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"--- 2. Loading {len(df)} Signals (Parallel) ---")
    
    # Worker for threading
    def load_signal(row):
        try:
            fpath = os.path.join(data_root, row['filename_lr'])
            sig = wfdb.rdrecord(fpath).p_signal
            # Z-Score Norm
            sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
            return sig
        except:
            return None

    with ThreadPoolExecutor() as executor:
        signals = list(executor.map(load_signal, [r for _, r in df.iterrows()]))
    
    # Filter broken files
    X, y = [], []
    for s, label in zip(signals, df['target']):
        if s is not None and s.shape == (1000, 12):
            X.append(s)
            y.append(label)
            
    return np.array(X), np.array(y)

# ==============================================================================
# METHOD 1: STATISTICAL FEATURE ENGINEERING + XGBOOST
# ==============================================================================
def extract_stat_features(X):
    print("--- Extracting Statistical Features ---")
    # X shape: (N, 1000, 12)
    features = []
    
    # Vectorized operations are faster than loops
    mean = np.mean(X, axis=1) # (N, 12)
    std = np.std(X, axis=1)
    max_val = np.max(X, axis=1)
    min_val = np.min(X, axis=1)
    ptp = np.ptp(X, axis=1)   # Peak-to-peak
    skew = stats.skew(X, axis=1)
    kurt = stats.kurtosis(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    
    # Concatenate all features: 12 leads * 8 stats = 96 features
    features = np.hstack([mean, std, max_val, min_val, ptp, skew, kurt, rms])
    return features

def run_method_xgboost(X_train, y_train, X_test, y_test):
    print("\n[METHOD 1] Training XGBoost (Statistical)...")
    
    X_train_feats = extract_stat_features(X_train)
    X_test_feats = extract_stat_features(X_test)
    
    model = XGBClassifier(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=6, 
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train_feats, y_train)
    
    preds = model.predict_proba(X_test_feats)[:, 1]
    return preds, model

# ==============================================================================
# METHOD 2: INCEPTION-1D (MULTI-SCALE TEMPORAL)
# ==============================================================================
def inception_module(x, filters):
    # Parallel convolutions with different kernel sizes
    k1 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    k2 = layers.Conv1D(filters, 11, padding='same', activation='relu')(x)
    k3 = layers.Conv1D(filters, 21, padding='same', activation='relu')(x)
    
    # Max Pooling branch
    mp = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    mp = layers.Conv1D(filters, 1, padding='same', activation='relu')(mp)
    
    x = layers.Concatenate()([k1, k2, k3, mp])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def run_method_inception(X_train, y_train, X_test):
    print("\n[METHOD 2] Training Inception1D (Multi-Scale)...")
    
    inp = layers.Input(shape=(1000, 12))
    x = inception_module(inp, 32)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = inception_module(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inp, out, name="Inception1D")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0, callbacks=[callbacks.EarlyStopping(patience=5)])
    return model.predict(X_test).flatten()

# ==============================================================================
# METHOD 3: SPATIAL LEAD-ATTENTION NETWORK
# ==============================================================================
def run_method_spatial_attention(X_train, y_train, X_test):
    print("\n[METHOD 3] Training Spatial Lead-Attention...")
    
    # Input: (1000, 12) -> Transpose to (12, 1000) to treat Leads as Sequence
    X_train_T = np.transpose(X_train, (0, 2, 1))
    X_test_T = np.transpose(X_test, (0, 2, 1))
    
    inp = layers.Input(shape=(12, 1000))
    
    # Feature extraction per lead (Time Distributed)
    # We reduce 1000 timepoints to a smaller embedding vector per lead
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(inp) # (12, 128)
    
    # Self-Attention across Leads (Learning Spatial Correlation)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x) # Pool across leads
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inp, out, name="SpatialAttn")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    model.fit(X_train_T, y_train, epochs=20, batch_size=64, verbose=0, callbacks=[callbacks.EarlyStopping(patience=5)])
    return model.predict(X_test_T).flatten()

# ==============================================================================
# METHOD 4: FREQUENCY DOMAIN CNN (SPECTRALNET)
# ==============================================================================
def run_method_spectral(X_train, y_train, X_test):
    print("\n[METHOD 4] Training SpectralNet (Frequency Domain)...")
    
    # FFT Transform
    # Take absolute value of Real FFT (only positive freqs)
    X_train_fft = np.abs(np.fft.rfft(X_train, axis=1))
    X_test_fft = np.abs(np.fft.rfft(X_test, axis=1))
    
    # Log scale to dampen high energy peaks
    X_train_fft = np.log(X_train_fft + 1e-6)
    X_test_fft = np.log(X_test_fft + 1e-6)
    
    # Shape changes from (1000) -> (501)
    inp = layers.Input(shape=(X_train_fft.shape[1], 12))
    
    x = layers.Conv1D(32, 5, activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inp, out, name="SpectralNet")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    model.fit(X_train_fft, y_train, epochs=20, batch_size=64, verbose=0, callbacks=[callbacks.EarlyStopping(patience=5)])
    return model.predict(X_test_fft).flatten()

# ==============================================================================
# METHOD 5: ROCKET-LITE (RANDOM CONVOLUTIONAL KERNELS)
# ==============================================================================
def generate_random_kernels(n_kernels=2000, length=9, input_channels=12):
    weights = np.random.normal(0, 1, (n_kernels, length, input_channels))
    # Biases
    biases = np.random.uniform(-1, 1, n_kernels)
    return weights, biases

def apply_rocket(X, kernels, biases):
    # Simulating convolution via sliding window is slow in pure python
    # We will use a simplified global pooling version for speed (Random Projection)
    # Project: (N, Time, 12) * (12, kernels) -> Flattening time for speed
    
    # Faster approximate: Mean and Max over time, then random project
    # This captures statistical distribution but linearly
    
    # Correct Rocket implementation requires numba/C++. 
    # Here we use a "Random Projection" proxy which is vectorized.
    
    # 1. Flatten time: (N, 12000)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)
    
    # 2. Random Matrix: (12000, 2000)
    # We use a fixed random seed for reproducibility
    np.random.seed(42)
    proj_matrix = np.random.normal(0, 0.01, (X_flat.shape[1], 1024))
    
    # 3. Project
    features = np.dot(X_flat, proj_matrix)
    
    # 4. Non-linearity
    features = np.maximum(features, 0)
    return features

def run_method_rocket(X_train, y_train, X_test):
    print("\n[METHOD 5] Training Rocket-Lite (Random Projection)...")
    
    X_train_proj = apply_rocket(X_train, None, None)
    X_test_proj = apply_rocket(X_test, None, None)
    
    # Ridge Classifier is standard for Rocket
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_proj, y_train)
    
    # Ridge doesn't output probas easily, we use decision function
    preds = clf.decision_function(X_test_proj)
    # Sigmoid normalization
    preds = 1 / (1 + np.exp(-preds))
    return preds

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    # 1. Load
    X, y = load_and_prep_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"\nData Shape: {X_train.shape}")
    
    # 2. Run All Methods
    p1, _ = run_method_xgboost(X_train, y_train, X_test, y_test)
    p2 = run_method_inception(X_train, y_train, X_test)
    p3 = run_method_spatial_attention(X_train, y_train, X_test)
    p4 = run_method_spectral(X_train, y_train, X_test)
    p5 = run_method_rocket(X_train, y_train, X_test)
    
    # 3. Ensemble (Average)
    ensemble_preds = (p1 + p2 + p3 + p4 + p5) / 5.0
    
    # 4. Evaluation
    def evaluate(name, preds, true):
        auc = roc_auc_score(true, preds)
        acc = accuracy_score(true, (preds > 0.5).astype(int))
        return auc, acc

    results = {}
    results['XGBoost (Stats)'] = evaluate('XGB', p1, y_test)
    results['Inception1D'] = evaluate('Inc', p2, y_test)
    results['SpatialAttn'] = evaluate('Spat', p3, y_test)
    results['SpectralNet'] = evaluate('Freq', p4, y_test)
    results['RocketLite'] = evaluate('Rock', p5, y_test)
    results['ENSEMBLE'] = evaluate('ENS', ensemble_preds, y_test)
    
    print("\n" + "="*40)
    print("FINAL GRANDMASTER RESULTS")
    print("="*40)
    print(f"{'Method':<20} | {'AUC':<8} | {'Accuracy':<8}")
    print("-" * 40)
    
    for k, v in results.items():
        print(f"{k:<20} | {v[0]:.4f}   | {v[1]:.4f}")
        
    # Save Ensemble
    ensemble_binary = (ensemble_preds > 0.5).astype(int)
    cm = confusion_matrix(y_test, ensemble_binary)
    
    report = f"""
    FINAL ENSEMBLE METRICS:
    AUC: {results['ENSEMBLE'][0]}
    ACC: {results['ENSEMBLE'][1]}
    
    CONFUSION MATRIX:
    {cm}
    """
    
    with open(f"{CONFIG['OUTPUT_DIR']}/final_grandmaster_report.txt", "w") as f:
        f.write(report)
        
    print("\nPipeline Complete. Results saved.")

if __name__ == "__main__":
    main()