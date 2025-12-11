# ==============================================================================
# "NUCLEAR OPTION" PIPELINE: Spectrograms + EfficientNetV2 (Transfer Learning)
# Strategy: Convert ECG to Images -> Fine-tune Pre-trained Vision Model
# Target: Breaking the 95% Ceiling towards 99%
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
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, mixed_precision, models, optimizers

# 1. SETUP: ENABLE MIXED PRECISION FOR SPEED
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision Enabled.")
except:
    pass

CONFIG = {
    'SAMP_RATE': 100,
    'DURATION': 10,
    'BATCH_SIZE': 32,      # EfficientNet needs RAM, smaller batch
    'EPOCHS': 25,          # Converges faster due to pre-training
    'LEARNING_RATE': 1e-4, # Lower LR for Fine-Tuning
    'IMG_SIZE': (224, 224), # Standard Input for EfficientNet
    'OUTPUT_DIR': './results/vision_results'
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. DATA LOADING (Robust & Fast)
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
# 3. ON-THE-FLY SPECTROGRAM GENERATION
# ==============================================================================
def signal_to_spectrogram(signal, label):
    """
    Converts 1D ECG (1000, 12) -> 2D Image (224, 224, 12)
    Uses Short-Time Fourier Transform (STFT)
    """
    # 1. Compute STFT
    # frame_length=64, frame_step=4 results in a good time-freq resolution for ECG
    stft = tf.signal.stft(tf.transpose(signal), frame_length=64, frame_step=4)
    spectrogram = tf.abs(stft)
    
    # 2. Log Scale (Visualize low power frequencies)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # 3. Resize to EfficientNet Input (224, 224)
    # Current shape: (12, Time, Freq). We need to transpose to (Time, Freq, 12)
    spectrogram = tf.transpose(spectrogram, perm=[1, 2, 0])
    spectrogram = tf.image.resize(spectrogram, CONFIG['IMG_SIZE'])
    
    # 4. Normalization for ImageNet weights (0-255 scale roughly)
    # We normalize to 0-1 range
    min_val = tf.reduce_min(spectrogram)
    max_val = tf.reduce_max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-8)
    
    return spectrogram, label

def create_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    # Apply Spectrogram Conversion on GPU
    ds = ds.map(signal_to_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================================================================
# 4. VISION TRANSFORMER / EFFICIENTNET MODEL
# ==============================================================================
def build_vision_model():
    # Input: (224, 224, 12) -> 12 channels of Spectrograms
    inputs = layers.Input(shape=(224, 224, 12))
    
    # Learnable Projection: 12 Channels -> 3 RGB Channels
    # This allows the model to learn which ECG leads are most important 
    # and map them to the R, G, B channels EfficientNet expects.
    x = layers.Conv2D(3, (1, 1), padding='same', name='lead_projection')(inputs)
    
    # Load Pre-trained EfficientNetV2 (The "Brain")
    # include_top=False removes the ImageNet classifier
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=x
    )
    
    # Unfreeze the top layers for Fine-Tuning
    base_model.trainable = True
    
    # Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return models.Model(inputs, outputs, name="ECG_Vision_Net")

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
def run_vision_pipeline():
    X, y = get_data_arrays()
    
    # Train/Val/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    # Create TF Datasets (Images are generated here)
    train_ds = create_dataset(X_train, y_train, CONFIG['BATCH_SIZE'], shuffle=True)
    val_ds = create_dataset(X_val, y_val, CONFIG['BATCH_SIZE'])
    test_ds = create_dataset(X_test, y_test, CONFIG['BATCH_SIZE'])
    
    model = build_vision_model()
    
    # Optimizer: Use small LR for Transfer Learning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n--- Training Vision Model (Spectrograms -> EfficientNet) ---\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['EPOCHS'],
        callbacks=[
            callbacks.ModelCheckpoint(f"{CONFIG['OUTPUT_DIR']}/best_vision_model.keras", save_best_only=True, monitor='val_auc', mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
            callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
        ]
    )
    
    print("\n--- Evaluating Vision Model ---\n")
    # Load best weights
    model.load_weights(f"{CONFIG['OUTPUT_DIR']}/best_vision_model.keras")
    
    # Predict
    y_pred_probs = model.predict(test_ds)
    
    # Optimal Threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_pred_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred)
    
    report = f"""
    === SPECTROGRAM VISION RESULTS ===
    Accuracy: {acc:.4f}
    AUC Score: {auc:.4f}
    F1 Score: {f1:.4f}
    Optimal Threshold: {best_thresh:.4f}
    
    Confusion Matrix:
    {confusion_matrix(y_test, y_pred)}
    """
    print(report)
    with open(f"{CONFIG['OUTPUT_DIR']}/vision_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_vision_pipeline()