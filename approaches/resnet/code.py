# ==============================================================================
# ECG Myocardial Infarction Detection Research Pipeline (2025 Standard)
# Dataset: PTB-XL (via KaggleHub)
# Model: Hybrid 1D-ResNet with Multi-Head Attention (State-of-the-Art Architecture)
# Author: Gemini Research Bot
# ==============================================================================

import ast
import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================
CONFIG = {
    'SAMPLING_RATE': 100,      # Using 100Hz data for computational efficiency (sufficient for MI)
    'DURATION': 10,            # Seconds
    'LEADS': 12,
    'BATCH_SIZE': 64,
    'EPOCHS': 30,             # Adjust based on Colab limits
    'LEARNING_RATE': 0.001,
    'DATA_PATH': None,         # Will be set by kagglehub
    'OUTPUT_DIR': './results/resnet'
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

def install_dependencies():
    print("Checking dependencies...")
    try:
        import wfdb
    except ImportError:
        print("Installing wfdb...")
        os.system('pip install wfdb')
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system('pip install kagglehub')

install_dependencies()

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def find_dataset_root(base_path, filename='ptbxl_database.csv'):
    """Recursively searches for the database file to handle variable directory structures."""
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            print(f"Found {filename} in: {root}")
            return root
    raise FileNotFoundError(f"Could not find {filename} in {base_path} or its subdirectories.")

def load_ptb_xl():
    print("--- Downloading/Loading PTB-XL Dataset via KaggleHub ---")
    # This downloads the dataset to a local cache
    # Using the specific version or dataset handle
    download_path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    print(f"Dataset downloaded to: {download_path}")
    
    # FIX: Robustly find the directory containing the CSV
    actual_data_path = find_dataset_root(download_path, 'ptbxl_database.csv')
    CONFIG['DATA_PATH'] = actual_data_path
    
    # Load Metadata
    Y = pd.read_csv(os.path.join(actual_data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    
    # Preprocessing labels: SCP Codes are stored as string dictionaries
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load SCP statements (mapping codes to classes)
    agg_df = pd.read_csv(os.path.join(actual_data_path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply mapping to get superclasses (NORM, MI, STTC, CD, HYP)
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    
    # Filter: We only want MI (Myocardial Infarction) vs NORM (Normal)
    print("Filtering dataset for MI vs NORM...")
    
    mask_mi = Y['diagnostic_superclass'].apply(lambda x: 'MI' in x)
    mask_norm = Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x)
    
    # We select samples that are EITHER MI OR NORM
    Y_mi = Y[mask_mi].copy()
    Y_mi['target'] = 1
    Y_norm = Y[mask_norm].copy()
    Y_norm['target'] = 0
    
    # Downsample Normals to balance dataset 
    # (Optional: remove this block if you want to test on imbalanced data)
    if len(Y_norm) > len(Y_mi):
        print(f"Balancing classes: Downsampling NORM from {len(Y_norm)} to {len(Y_mi)}")
        Y_norm = Y_norm.sample(n=len(Y_mi), random_state=42)
        
    df = pd.concat([Y_mi, Y_norm])
    
    print(f"Selected Data: {len(df)} samples. (MI: {len(Y_mi)}, NORM: {len(Y_norm)})")
    
    return df, actual_data_path

def load_raw_signals(df, path):
    print("--- Loading Raw Binary Signals (This may take a moment) ---")
    data = []
    
    cnt = 0
    for idx, row in df.iterrows():
        # Use Low Res (100Hz) for efficiency.
        filename = row['filename_lr'] 
        
        # Ensure path is relative to the found data root
        file_path = os.path.join(path, filename)
        
        try:
            # wfdb reads both .dat and .hea
            record = wfdb.rdrecord(file_path)
            signal = record.p_signal
            
            # Standardization (Z-score normalization per lead)
            # Crucial for Neural Networks to converge
            signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
            
            data.append(signal)
        except Exception as e:
            # Try removing leading slash if present in filename string
            try:
                if filename.startswith('/'):
                    filename = filename[1:]
                file_path = os.path.join(path, filename)
                record = wfdb.rdrecord(file_path)
                signal = record.p_signal
                signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
                data.append(signal)
            except:
                print(f"Error reading {file_path}: {e}")
            
        cnt += 1
        if cnt % 1000 == 0:
            print(f"Loaded {cnt} signals...")
            
    return np.array(data)

# ==============================================================================
# 3. MODEL ARCHITECTURE: 1D-ResNet + Attention
# ==============================================================================

def residual_block(x, filters, kernel_size=5, stride=1):
    shortcut = x
    
    # First Conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Second Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut handling (if dimensions change)
    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # --- Feature Extraction (ResNet Backbone) ---
    x = layers.Conv1D(32, 15, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = residual_block(x, 32, 5, 2) # Downsample
    x = residual_block(x, 64, 5, 2)
    x = residual_block(x, 128, 3, 2)
    x = residual_block(x, 256, 3, 2)
    
    # --- Attention Mechanism (The Breakthrough Part) ---
    # We treat the time steps as a sequence and apply Multi-Head Attention
    # This allows the model to look at the whole signal (global context)
    
    # Positional encoding is implicit in CNN, but MHSA needs sequence dimension
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention_output]) # Residual connection around attention
    x = layers.LayerNormalization()(x)
    
    # --- Classification Head ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x) # Binary Classification
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_Attention_ECG")
    return model

# ==============================================================================
# 4. EXECUTION PIPELINE
# ==============================================================================

def run_pipeline():
    # 1. Load Data
    try:
        df_labels, data_path = load_ptb_xl()
        X = load_raw_signals(df_labels, data_path)
        y = df_labels['target'].values
        
        # Check if data loaded correctly
        if len(X) == 0:
            print("CRITICAL ERROR: No signals were loaded. Check paths.")
            return

        print(f"Final Dataset Shape: X={X.shape}, y={y.shape}")
        
        # 2. Split Data (Train/Val/Test)
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
        
        # 3. Build Model
        input_shape = (X_train.shape[1], X_train.shape[2]) # (1000, 12)
        model = build_model(input_shape)
        
        model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        model.summary()
        
        # 4. Train
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['EPOCHS'],
            batch_size=CONFIG['BATCH_SIZE'],
            callbacks=callbacks_list
        )
        
        # 5. Evaluate & Save Results
        print("--- Evaluating Model ---")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Report String
        report = f"""
        --- MI Detection Research Report ---
        Model: Hybrid ResNet-1D + MultiHead Attention
        Dataset: PTB-XL (Balanced Subset)
        
        Accuracy: {acc:.4f}
        AUC Score: {auc:.4f}
        F1 Score: {f1:.4f}
        
        Confusion Matrix:
        {cm}
        
        Classification Report:
        {classification_report(y_test, y_pred)}
        """
        print(report)
        
        with open(f"{CONFIG['OUTPUT_DIR']}/final_report.txt", "w") as f:
            f.write(report)
            
        # Save Model
        model.save(f"{CONFIG['OUTPUT_DIR']}/mi_detection_model.keras")
        
        # Plot History
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.legend()
        plt.title('AUC')
        plt.savefig(f"{CONFIG['OUTPUT_DIR']}/training_curves.png")
        plt.show() # Show in notebook
        
        print(f"All results saved to {CONFIG['OUTPUT_DIR']}")
        
    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()