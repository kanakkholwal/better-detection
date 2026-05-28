# ==============================================================================
# LGS-NET: Local-Global-Spectral Attention Network for MI Detection
# Dataset: Full PTB-XL (21,837 patients)
# Architecture: Dual-Stream (Raw Signal + STFT) with Local Window Attention
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

# 1. PRODUCTION SETUP: MIXED PRECISION & GPU
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✓ Mixed Precision Enabled (FP16).")
except:
    pass

CONFIG = {
    'SAMP_RATE': 100,
    'DURATION': 10,
    'BATCH_SIZE': 64,      
    'EPOCHS': 40,          
    'LEARNING_RATE': 1e-3, 
    'IMG_SIZE': (128, 64), # Optimal for STFT features
    'WINDOW_SIZE': 50,     # Local Attention Window (500ms - covers 1 beat roughly)
    'OUTPUT_DIR': './lgs_net_results'
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. HIGH-PERFORMANCE DATA PIPELINE
# ==============================================================================
def find_dataset_root(base_path, filename='ptbxl_database.csv'):
    for root, _, files in os.walk(base_path):
        if filename in files: return root
    raise FileNotFoundError("Database CSV not found.")

def load_signal_worker(args):
    """ Worker for ThreadPoolExecutor """
    file_path, target_len = args
    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal
        # Robust Z-Score Normalization
        signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
        # Pad/Crop
        if len(signal) > target_len:
            signal = signal[:target_len]
        elif len(signal) < target_len:
            pad = np.zeros((target_len - len(signal), 12))
            signal = np.vstack([signal, pad])
        return signal
    except:
        return None

def get_data_arrays():
    print("--- 1. Initializing Data Ingestion ---")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    data_path = find_dataset_root(path)
    
    # Load Labels
    Y = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index: tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    
    # Binary Classification: MI vs NORM
    Y_mi = Y[Y['diagnostic_superclass'].apply(lambda x: 'MI' in x)].copy()
    Y_norm = Y[Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x)].copy()
    
    Y_mi['target'] = 1
    # Balance majority class
    if len(Y_norm) > len(Y_mi):
        Y_norm = Y_norm.sample(n=len(Y_mi), random_state=42)
    Y_norm['target'] = 0
    
    df = pd.concat([Y_mi, Y_norm]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    target_len = CONFIG['DURATION'] * CONFIG['SAMP_RATE']
    file_paths = [os.path.join(data_path, row['filename_lr']) for _, row in df.iterrows()]
    
    print(f"--- 2. Threaded Loading of {len(file_paths)} Signals ---")
    X = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(load_signal_worker, [(p, target_len) for p in file_paths])
    
    valid_indices = []
    for idx, res in enumerate(results):
        if res is not None:
            X.append(res)
            valid_indices.append(idx)
            
    X = np.array(X, dtype=np.float32)
    y = df.iloc[valid_indices]['target'].values.astype(np.float32)
    return X, y

# ==============================================================================
# 3. DUAL-INPUT GENERATOR (Raw + STFT)
# ==============================================================================
def preprocess_dual_input(signal, label):
    # signal: (1000, 12)

    # STFT per lead
    stft = tf.signal.stft(
        tf.transpose(signal),      # (12, time)
        frame_length=64,
        frame_step=8
    )                               # (12, frames, freq)

    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)

    # Move leads to channel dim
    spectrogram = tf.transpose(spectrogram, perm=[1, 2, 0])
    # (frames, freq, 12)

    # Resize spatial dims ONLY
    spectrogram = tf.image.resize(
        spectrogram,
        CONFIG['IMG_SIZE']
    )  
    # (128, 64, 12)

    # Normalize
    spectrogram = tf.cast(spectrogram, tf.float32)
    min_val = tf.reduce_min(spectrogram)
    max_val = tf.reduce_max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-8)

    return {
        'raw_input': signal,
        'stft_input': spectrogram
    }, label


def create_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(preprocess_dual_input, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================================================================
# 4. CUSTOM LAYERS: THE INNOVATION
# ==============================================================================
class LocalWindowAttention(layers.Layer):
    """
    Applies Self-Attention locally within small time windows.
    Detects local morphological changes (ST elevation) regardless of global position.
    """
    def __init__(self, window_size, embed_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs):
        # inputs shape: (Batch, Time, Channels)
        B, T, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Pad if Time is not divisible by Window Size
        pad_amt = (self.window_size - (T % self.window_size)) % self.window_size
        x = tf.pad(inputs, [[0, 0], [0, pad_amt], [0, 0]])
        
        # Reshape to (Batch * Num_Windows, Window_Size, Channels)
        # This forces attention to only look at 'Window_Size' neighbors
        num_windows = tf.shape(x)[1] // self.window_size
        x_windows = tf.reshape(x, (-1, self.window_size, C))
        
        # Apply Self-Attention locally
        att_out = self.att(x_windows, x_windows)
        x_windows = self.norm(x_windows + att_out)
        
        # Reshape back to original sequence
        x_out = tf.reshape(x_windows, (B, -1, C))
        
        # Remove padding
        return x_out[:, :T, :]

class LeadWiseAttention(layers.Layer):
    """
    Channel-wise Squeeze-and-Excitation.
    Works for ANY channel size (12, 64, 128, ...).
    """
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction)

        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(reduced, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        # inputs: (B, T, C)
        se = self.gap(inputs)          # (B, C)
        se = self.fc1(se)              # (B, C//r)
        se = self.fc2(se)              # (B, C)
        se = tf.expand_dims(se, 1)     # (B, 1, C)
        return inputs * se


# ==============================================================================
# 5. MODEL ARCHITECTURE: LGS-NET
# ==============================================================================
def build_lgs_net():
    # --- BRANCH 1: TIME DOMAIN (Local Attention) ---
    raw_in = layers.Input(shape=(1000, 12), name='raw_input')
    
    # 1. Feature Extraction (ResNet-style)
    x1 = layers.Conv1D(64, 7, padding='same', activation='relu')(raw_in)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(2)(x1) # 500
    
    x1 = layers.Conv1D(128, 5, padding='same', activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(2)(x1) # 250
    
    # 2. Local Window Attention (The Innovation)
    # Window size 25 corresponds to ~250ms at this depth (covering 1 beat feature)
    x1 = LocalWindowAttention(window_size=25, embed_dim=128, num_heads=4)(x1)
    
    # 3. Lead-Wise Spatial Attention
    x1 = LeadWiseAttention()(x1)
    
    x1 = layers.GlobalAveragePooling1D()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # --- BRANCH 2: FREQUENCY DOMAIN (Spectral) ---
    stft_in = layers.Input(shape=(128, 64, 12), name='stft_input')
    
    # Standard 2D CNN for Spectrograms
    x2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(stft_in)
    x2 = layers.MaxPooling2D((2,2))(x2)
    x2 = layers.BatchNormalization()(x2)
    
    x2 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D((2,2))(x2)
    x2 = layers.BatchNormalization()(x2)
    
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dropout(0.3)(x2)
    
    # --- FUSION ---
    combined = layers.Concatenate()([x1, x2])
    
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation='relu')(z)
    
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(z)
    
    return models.Model(inputs=[raw_in, stft_in], outputs=outputs, name="LGS_Net")

# ==============================================================================
# 6. EXECUTION
# ==============================================================================
def run_lgs_pipeline():
    # 1. Load Data
    X, y = get_data_arrays()
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    print(f"Training Samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 3. Pipelines
    train_ds = create_dataset(X_train, y_train, CONFIG['BATCH_SIZE'], shuffle=True)
    val_ds = create_dataset(X_val, y_val, CONFIG['BATCH_SIZE'])
    test_ds = create_dataset(X_test, y_test, CONFIG['BATCH_SIZE'])
    
    # 4. Build & Compile
    model = build_lgs_net()
    
    # AdamW is critical for Attention convergence
    optimizer = optimizers.AdamW(learning_rate=CONFIG['LEARNING_RATE'], weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # 5. Train
    print("\n--- Training LGS-Net (Local-Global-Spectral) ---\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['EPOCHS'],
        callbacks=[
            callbacks.ModelCheckpoint(
                f"{CONFIG['OUTPUT_DIR']}/best_lgs_model.keras",
                save_best_only=True,
                monitor='val_auc',
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]

    )
    
    # 6. Evaluate
    print("\n--- Final Evaluation ---\n")
    model.load_weights(f"{CONFIG['OUTPUT_DIR']}/best_lgs_model.keras")
    
    # Predict (Must iterate over dataset to get y_true aligned if strictly needed, 
    # but sklearn metrics need arrays. We predict on DS and use y_test array)
    y_pred_probs = model.predict(test_ds)
    
    # Optimal Threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_pred_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    
    report = f"""
    === LGS-NET FINAL RESULTS ===
    Architecture: Dual-Stream (ResNet-Attention + STFT-CNN)
    Attention Mode: Local Window + Lead-Wise
    
    Accuracy: {acc:.4f}
    AUC Score: {auc:.4f}
    Optimal Threshold: {best_thresh:.4f}
    
    Confusion Matrix:
    {confusion_matrix(y_test, y_pred)}
    """
    print(report)
    with open(f"{CONFIG['OUTPUT_DIR']}/lgs_net_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_lgs_pipeline()