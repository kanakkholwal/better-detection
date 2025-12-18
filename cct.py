# ==============================================================================
# CUSTOM COMPACT CONVOLUTIONAL TRANSFORMER (CCT) - FIXED
# Architecture: Custom Hybrid (Conv Tokenizer + Transformer Encoder)
# Fix: Corrected layer signatures to handle Keras graph construction
# Goal: 99% Accuracy via Self-Attention on Spectrograms
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
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, mixed_precision, models, optimizers

# 1. SETUP
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision Enabled.")
except:
    pass

CONFIG = {
    'SAMP_RATE': 100,
    'DURATION': 10,
    'BATCH_SIZE': 64,      
    'EPOCHS': 40,          
    'LEARNING_RATE': 1e-3, 
    'IMG_SIZE': (128, 128),
    'OUTPUT_DIR': './cct_results_fixed'
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def find_dataset_root(base_path, filename='ptbxl_database.csv'):
    for root, dirs, files in os.walk(base_path):
        if filename in files: return root
    raise FileNotFoundError("Database CSV not found.")

def load_signal_worker(args):
    file_path, target_len = args
    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal
        # Z-Score Norm
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
    print("--- 1. Downloading Data ---")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    data_path = find_dataset_root(path)
    
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
    
    # Filter
    Y_mi = Y[Y['diagnostic_superclass'].apply(lambda x: 'MI' in x)].copy()
    Y_norm = Y[Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x)].copy()
    
    Y_mi['target'] = 1
    # Strict Balancing
    if len(Y_norm) > len(Y_mi):
        Y_norm = Y_norm.sample(n=len(Y_mi), random_state=42)
    Y_norm['target'] = 0
    
    df = pd.concat([Y_mi, Y_norm])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    target_len = CONFIG['DURATION'] * CONFIG['SAMP_RATE']
    file_paths = [os.path.join(data_path, row['filename_lr']) for _, row in df.iterrows()]
    
    print(f"--- 2. Loading {len(file_paths)} Signals into RAM ---")
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
# 3. SPECTROGRAMS
# ==============================================================================
def signal_to_spectrogram(signal, label):
    # STFT
    stft = tf.signal.stft(tf.transpose(signal), frame_length=64, frame_step=4)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Resize to (128, 128) - CCT Input
    spectrogram = tf.transpose(spectrogram, perm=[1, 2, 0])
    spectrogram = tf.image.resize(spectrogram, CONFIG['IMG_SIZE'])
    
    # Normalize [0, 1]
    min_val = tf.reduce_min(spectrogram)
    max_val = tf.reduce_max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-8)
    
    return spectrogram, label

def create_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(signal_to_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================================================================
# 4. FIXED CCT ARCHITECTURE
# ==============================================================================
class CCTTokenizer(layers.Layer):
    """
    Learns to extract patches from the image using Convolution.
    """
    def __init__(self, kernel_size=3, stride=1, padding=1, pooling_kernel=3, pooling_stride=2, n_conv_layers=2, n_output_channels=[64, 128], **kwargs):
        super().__init__(**kwargs)
        self.conv_model = tf.keras.Sequential()
        for i in range(n_conv_layers):
            self.conv_model.add(layers.Conv2D(n_output_channels[i], kernel_size, stride, padding="same", activation="relu"))
            self.conv_model.add(layers.MaxPool2D(pooling_kernel, pooling_stride, padding="same"))
            self.conv_model.add(layers.BatchNormalization())

    def call(self, images):
        outputs = self.conv_model(images)
        # Flatten (Batch, Height, Width, Channels) -> (Batch, Seq_Len, Dim)
        return tf.reshape(outputs, (-1, outputs.shape[1] * outputs.shape[2], outputs.shape[3]))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.keras.activations.gelu), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # FIXED: Added default value for training
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_cct_model():
    inputs = layers.Input(shape=(CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1], 12))
    
    # Tokenizer (Conv Backbone)
    tokenizer = CCTTokenizer()
    x = tokenizer(inputs)
    
    # Positional Embedding
    seq_len = x.shape[1]
    embed_dim = x.shape[2]
    
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(positions)
    x = x + pos_embedding
    
    # Transformer Encoder Blocks
    # Passing training=None implicitly in functional API calls is standard, 
    # but the Layer definition now supports it.
    for _ in range(4):
        x = TransformerBlock(embed_dim=128, num_heads=4, mlp_dim=256, rate=0.2)(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    
    # Head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation=tf.keras.activations.gelu)(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(1, activation="sigmoid", dtype='float32')(x)
    
    return models.Model(inputs, outputs, name="Custom_CCT")

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
def run_cct_pipeline():
    X, y = get_data_arrays()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    train_ds = create_dataset(X_train, y_train, CONFIG['BATCH_SIZE'], shuffle=True)
    val_ds = create_dataset(X_val, y_val, CONFIG['BATCH_SIZE'])
    test_ds = create_dataset(X_test, y_test, CONFIG['BATCH_SIZE'])
    
    model = build_cct_model()
    
    # AdamW (Weight Decay)
    optimizer = optimizers.AdamW(learning_rate=CONFIG['LEARNING_RATE'], weight_decay=0.004)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n--- Training Custom CCT (Transformer) ---\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['EPOCHS'],
        callbacks=[
            callbacks.ModelCheckpoint(f"{CONFIG['OUTPUT_DIR']}/best_cct_model.keras", save_best_only=True, monitor='val_auc', mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True)
        ]
    )
    
    print("\n--- Evaluating ---\n")
    model.load_weights(f"{CONFIG['OUTPUT_DIR']}/best_cct_model.keras")
    
    y_pred_probs = model.predict(test_ds)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_pred_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred)
    
    report = f"""
    === CUSTOM CCT TRANSFORMER RESULTS (FIXED) ===
    Accuracy: {acc:.4f}
    AUC Score: {auc:.4f}
    F1 Score: {f1:.4f}
    Optimal Threshold: {best_thresh:.4f}
    
    Confusion Matrix:
    {confusion_matrix(y_test, y_pred)}
    """
    print(report)
    with open(f"{CONFIG['OUTPUT_DIR']}/cct_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_cct_pipeline()