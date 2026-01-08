# ==============================================================================
# TITAN ECG PIPELINE: 5 High-Performance Deep Learning Methods
# Goal: 99% Accuracy via Ensembling Strong Learners (No Weak Learners)
# ==============================================================================

import ast
import os
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import (
    callbacks,
    layers,
    mixed_precision,
    models,
    optimizers,
    regularizers,
)
from xgboost import XGBClassifier

# 1. SETUP & OPTIMIZATION
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision Enabled (GPU Speedup).")
except:
    pass

CONFIG = {
    'SAMP_RATE': 100,
    'DURATION': 10,
    'BATCH_SIZE': 64,
    'EPOCHS': 40,
    'OUTPUT_DIR': './titan_results'
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. DATA LOADING (Robust & Fast)
# ==============================================================================
def load_data():
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
    agg_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def get_label(y_dic):
        tmp = [agg_df.loc[k].diagnostic_class for k in y_dic if k in agg_df.index]
        return list(set(tmp))

    Y['diagnostic'] = Y.scp_codes.apply(get_label)
    
    mi_df = Y[Y['diagnostic'].apply(lambda x: 'MI' in x)].copy()
    norm_df = Y[Y['diagnostic'].apply(lambda x: 'NORM' in x)].copy()
    
    mi_df['target'] = 1
    # Strict Balancing
    if len(norm_df) > len(mi_df):
        norm_df = norm_df.sample(n=len(mi_df), random_state=42)
    norm_df['target'] = 0
    
    df = pd.concat([mi_df, norm_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"--- 2. Loading {len(df)} Signals ---")
    
    def load_sig(row):
        try:
            fpath = os.path.join(data_root, row['filename_lr'])
            sig = wfdb.rdrecord(fpath).p_signal
            # Z-Score Norm
            sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
            return sig
        except: return None

    with ThreadPoolExecutor() as ex:
        signals = list(ex.map(load_sig, [r for _, r in df.iterrows()]))
        
    X, y = [], []
    for s, l in zip(signals, df['target']):
        if s is not None and s.shape == (1000, 12):
            X.append(s)
            y.append(l)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ==============================================================================
# METHOD A: INCEPTION-TIME V3 (Optimized)
# ==============================================================================
def inception_block(x, filters):
    # Bottleneck
    bottleneck = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    
    # Multiscale
    k1 = layers.Conv1D(filters, 3, padding='same', activation='relu')(bottleneck)
    k2 = layers.Conv1D(filters, 9, padding='same', activation='relu')(bottleneck)
    k3 = layers.Conv1D(filters, 19, padding='same', activation='relu')(bottleneck)
    
    # MaxPool branch
    mp = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    mp = layers.Conv1D(filters, 1, padding='same', activation='relu')(mp)
    
    x = layers.Concatenate()([k1, k2, k3, mp])
    x = layers.BatchNormalization()(x)
    return x

def build_inception():
    inp = layers.Input(shape=(1000, 12))
    x = inp
    
    # Residual Inception Blocks
    for _ in range(3):
        x_res = x
        x = inception_block(x, 32)
        x = layers.Dropout(0.2)(x)
        x = inception_block(x, 32)
        
        # Project residual if needed
        if x_res.shape[-1] != x.shape[-1]:
            x_res = layers.Conv1D(x.shape[-1], 1, padding='same')(x_res)
        x = layers.Add()([x, x_res])
        x = layers.Activation('relu')(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inp, out, name="InceptionV3")

# ==============================================================================
# METHOD B: RESNET-1D + SE BLOCKS (Squeeze & Excitation)
# ==============================================================================
def se_block(x, filters, ratio=16):
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

def resnet_se_block(x, filters, kernel=7):
    shortcut = layers.Conv1D(filters, 1, padding='same')(x)
    
    x = layers.Conv1D(filters, kernel, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Add SE
    x = se_block(x, filters)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_se():
    inp = layers.Input(shape=(1000, 12))
    x = layers.Conv1D(64, 15, padding='same', activation='relu')(inp)
    
    x = resnet_se_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = resnet_se_block(x, 128)
    x = layers.MaxPooling1D(2)(x)
    x = resnet_se_block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inp, out, name="ResNetSE")

# ==============================================================================
# METHOD C: DENSENET-1D
# ==============================================================================
def dense_block(x, filters):
    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv1D(filters, 5, padding='same')(x1)
    return layers.Concatenate()([x, x1])

def build_densenet():
    inp = layers.Input(shape=(1000, 12))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    
    for _ in range(4):
        x = dense_block(x, 16) # Growth rate 16
    x = layers.MaxPooling1D(2)(x)
    
    for _ in range(4):
        x = dense_block(x, 16)
    x = layers.GlobalAveragePooling1D()(x)
    
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inp, out, name="DenseNet")

# ==============================================================================
# METHOD D: BI-LSTM + ATTENTION
# ==============================================================================
def build_lstm_attn():
    inp = layers.Input(shape=(1000, 12))
    # Downsample first to save compute
    x = layers.Conv1D(64, 5, strides=2, activation='relu')(inp) 
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Attention
    attn = layers.Dense(1, activation='tanh')(x)
    attn = layers.Flatten()(attn)
    attn = layers.Activation('softmax')(attn)
    attn = layers.RepeatVector(128)(attn)
    attn = layers.Permute([2, 1])(attn)
    
    sent_representation = layers.Multiply()([x, attn])
    sent_representation = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(sent_representation)
    
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(sent_representation)
    return models.Model(inp, out, name="BiLSTM_Attn")

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def train_model(model_builder, X_train, y_train, X_test, y_test, name):
    print(f"\n--- Training {name} ---")
    model = model_builder()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    cb = [
        # callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2)
    ]
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'], 
              callbacks=cb, verbose=1)
    
    # Extract Embeddings if it's Inception (For Method E)
    if name == "InceptionV3":
        embedding_model = models.Model(model.input, model.layers[-2].output)
        return model, embedding_model
        
    return model, None

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    preds = {}
    
    # 1. Train Inception (Anchor)
    model_inc, embed_inc = train_model(build_inception, X_train, y_train, X_test, y_test, "InceptionV3")
    preds['Inception'] = model_inc.predict(X_test).flatten()
    
    # 2. Train ResNet-SE
    model_res, _ = train_model(build_resnet_se, X_train, y_train, X_test, y_test, "ResNetSE")
    preds['ResNetSE'] = model_res.predict(X_test).flatten()
    
    # 3. Train DenseNet
    model_dense, _ = train_model(build_densenet, X_train, y_train, X_test, y_test, "DenseNet")
    preds['DenseNet'] = model_dense.predict(X_test).flatten()
    
    # 4. Train LSTM-Attn
    model_lstm, _ = train_model(build_lstm_attn, X_train, y_train, X_test, y_test, "BiLSTM_Attn")
    preds['LSTM'] = model_lstm.predict(X_test).flatten()
    
    # 5. METHOD E: XGBOOST ON DEEP EMBEDDINGS
    print("\n--- Training Method E: XGBoost on Inception Embeddings ---")
    train_embeds = embed_inc.predict(X_train)
    test_embeds = embed_inc.predict(X_test)
    
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1, eval_metric='auc')
    xgb.fit(train_embeds, y_train)
    preds['XGB_Embed'] = xgb.predict_proba(test_embeds)[:, 1]
    
    # --- ENSEMBLE ---
    # Weighted Average: Give more weight to Inception and XGB
    final_pred = (
        0.3 * preds['Inception'] + 
        0.2 * preds['ResNetSE'] + 
        0.15 * preds['DenseNet'] + 
        0.15 * preds['LSTM'] + 
        0.2 * preds['XGB_Embed']
    )
    
    # Evaluation
    acc = accuracy_score(y_test, (final_pred > 0.5).astype(int))
    auc = roc_auc_score(y_test, final_pred)
    
    print("\n" + "="*40)
    print("TITAN PIPELINE RESULTS")
    print("="*40)
    for k, v in preds.items():
        print(f"{k:<15} | AUC: {roc_auc_score(y_test, v):.4f}")
    print("-" * 40)
    print(f"TITAN ENSEMBLE  | AUC: {auc:.4f} | ACC: {acc:.4f}")
    
    # Save
    cm = confusion_matrix(y_test, (final_pred > 0.5).astype(int))
    report = f"AUC: {auc}\nACC: {acc}\nCM:\n{cm}"
    with open(f"{CONFIG['OUTPUT_DIR']}/titan_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()