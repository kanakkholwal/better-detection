# ==============================================================================
# ECG MASTER PIPELINE - FAST VERSION
# Target: ≥96% Accuracy & AUC on PTB-XL Dataset
#
# pip install numpy pandas tensorflow scikit-learn imbalanced-learn xgboost lightgbm optuna kagglehub PyWavelet wfdb optuna-integration[tfkeras]

import ast

# ==============================================================================
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'PATIENCE': 8,
    'LEARNING_RATE': 0.001,
    'OUTPUT_DIR': './ecg_fast_results'
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. FAST DATA LOADING
# ==============================================================================

def load_data_fast():
    """Load PTB-XL dataset quickly without complex feature extraction"""
    print("📥 Loading PTB-XL dataset...")
    
    # Download dataset
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    
    # Find CSV files
    csv_path = None
    data_root = None
    for root, _, files in os.walk(path):
        if 'ptbxl_database.csv' in files:
            csv_path = os.path.join(root, 'ptbxl_database.csv')
            data_root = root
            break
    
    # Load metadata
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    
    # Load diagnostic statements
    scp_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]
    
    # Parse labels - MI (Myocardial Infarction) vs Normal
    def get_diagnostic_label(codes):
        diagnostic_classes = []
        for code in codes:
            if code in scp_df.index:
                diagnostic_classes.append(scp_df.loc[code].diagnostic_class)
        return list(set(diagnostic_classes))
    
    df['diagnostic'] = df.scp_codes.apply(get_diagnostic_label)
    
    # Create binary labels: 1 for MI, 0 for Normal
    df['label'] = df['diagnostic'].apply(
        lambda x: 1 if 'MI' in x else (0 if 'NORM' in x else -1)
    )
    
    # Filter only MI and Normal classes
    df = df[df['label'] != -1].reset_index(drop=True)
    
    print(f"📊 Dataset: {len(df)} samples (MI: {sum(df['label'] == 1)}, Normal: {sum(df['label'] == 0)})")
    
    # Load signals with simple preprocessing
    def load_signal(row):
        try:
            filepath = os.path.join(data_root, row['filename_lr'])
            signal_data = wfdb.rdrecord(filepath).p_signal
            
            # Ensure correct shape
            if signal_data.shape[0] != 1000:
                from scipy import signal as sp_signal
                signal_data = sp_signal.resample(signal_data, 1000, axis=0)
            
            # Simple normalization
            signal_data = (signal_data - np.mean(signal_data, axis=0)) / (np.std(signal_data, axis=0) + 1e-8)
            return signal_data
        except Exception as e:
            return None
    
    # Load signals in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        signals = list(executor.map(load_signal, [row for _, row in df.iterrows()]))
    
    # Filter failed loads
    X = []
    y = []
    for signal_data, label in zip(signals, df['label']):
        if signal_data is not None and signal_data.shape == (1000, 12):
            X.append(signal_data)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} signals")
    return X, y

def augment_data_simple(X, y):
    """Simple data augmentation"""
    print("🔄 Applying simple augmentation...")
    
    # Apply SMOTE
    smote = SMOTE(random_state=CONFIG['SEED'])
    X_reshaped = X.reshape(X.shape[0], -1)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    X_resampled = X_resampled.reshape(-1, 1000, 12)
    
    # Add noise for minority class
    X_augmented = []
    y_augmented = []
    
    for i, (signal_data, label) in enumerate(zip(X_resampled, y_resampled)):
        X_augmented.append(signal_data)
        y_augmented.append(label)
        
        if label == 1:  # Add one augmented version for MI
            noise = np.random.normal(0, 0.02, signal_data.shape)
            X_augmented.append(signal_data + noise)
            y_augmented.append(label)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    # Shuffle
    idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[idx]
    y_augmented = y_augmented[idx]
    
    print(f"📈 Augmented dataset: {len(X_augmented)} samples")
    return X_augmented, y_augmented

# ==============================================================================
# 3. EFFICIENT MODEL ARCHITECTURES
# ==============================================================================

def build_efficient_transformer():
    """Lightweight transformer for ECG"""
    inputs = layers.Input(shape=(1000, 12))
    
    # Initial projection
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    
    # Simplified transformer block
    for _ in range(2):
        # Self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            dropout=0.1
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ffn = layers.Dense(128, activation='relu')(x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(64)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientTransformer')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def build_hybrid_cnn():
    """Hybrid CNN with multiple kernel sizes"""
    inputs = layers.Input(shape=(1000, 12))
    
    # Multiple parallel convolutions
    conv3 = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv3 = layers.BatchNormalization()(conv3)
    
    conv7 = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    conv7 = layers.BatchNormalization()(conv7)
    
    conv15 = layers.Conv1D(32, 15, padding='same', activation='relu')(inputs)
    conv15 = layers.BatchNormalization()(conv15)
    
    # Concatenate and process
    x = layers.Concatenate()([conv3, conv7, conv15])
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='HybridCNN')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def build_resnet_simple():
    """Simple ResNet architecture"""
    def res_block(x, filters):
        shortcut = x
        
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    inputs = layers.Input(shape=(1000, 12))
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    
    x = res_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    
    x = res_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNetSimple')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# ==============================================================================
# 4. TRAINING FUNCTIONS
# ==============================================================================

def train_model(model_builder, X_train, y_train, X_val, y_val, model_name):
    """Train a single model with verbose output"""
    print(f"\n🔧 Training {model_name}...")
    print("-" * 50)
    
    model = model_builder()
    
    callbacks_list = [
        # callbacks.EarlyStopping(
        #     monitor='val_auc',
        #     patience=CONFIG['PATIENCE'],
        #     restore_best_weights=True,
        #     mode='max',
        #     verbose=1
        # ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=CONFIG['PATIENCE'] // 2,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
    ]
    
    # Train with verbose output
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Get best validation AUC
    best_epoch = np.argmax(history.history['val_auc'])
    best_auc = history.history['val_auc'][best_epoch]
    best_acc = history.history['val_accuracy'][best_epoch]
    
    print(f"\n✅ {model_name} - Best epoch: {best_epoch + 1}")
    print(f"   Validation AUC: {best_auc:.4f}, Accuracy: {best_acc:.4f}")
    
    return model, history

def train_xgboost_on_features(X_train, y_train, X_val, y_val):
    """Train XGBoost on statistical features"""
    print("\n🌳 Training XGBoost on statistical features...")
    print("-" * 50)
    
    # Extract simple statistical features
    def extract_simple_features(X):
        features = []
        for signal in X:
            signal_features = []
            for lead in range(12):
                lead_signal = signal[:, lead]
                signal_features.extend([
                    np.mean(lead_signal),
                    np.std(lead_signal),
                    np.max(lead_signal),
                    np.min(lead_signal),
                    np.median(lead_signal),
                    np.percentile(lead_signal, 25),
                    np.percentile(lead_signal, 75)
                ])
            features.append(signal_features)
        return np.array(features)
    
    X_train_feats = extract_simple_features(X_train)
    X_val_feats = extract_simple_features(X_val)
    
    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=CONFIG['SEED'],
        eval_metric='auc',
        verbosity=1
    )
    
    xgb.fit(
        X_train_feats, y_train,
        eval_set=[(X_val_feats, y_val)],
        verbose=1
    )
    
    # Get predictions
    y_pred = xgb.predict_proba(X_val_feats)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))
    
    print(f"\n✅ XGBoost - Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    return xgb

# ==============================================================================
# 5. MAIN PIPELINE
# ==============================================================================

def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("🚀 ECG FAST PIPELINE - Starting Execution")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 STEP 1: Loading data...")
    X, y = load_data_fast()
    
    # Step 2: Split data
    print("\n✂️ STEP 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=CONFIG['SEED']
    )
    
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Step 3: Augment data
    print("\n🔄 STEP 3: Augmenting data...")
    X_train_aug, y_train_aug = augment_data_simple(X_train, y_train)
    
    # Step 4: Create validation split
    print("\n🎯 STEP 4: Creating validation split...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_aug, y_train_aug, test_size=0.15, stratify=y_train_aug, random_state=CONFIG['SEED']
    )
    
    print(f"  Final Training: {X_train_final.shape[0]} samples")
    print(f"  Validation:     {X_val.shape[0]} samples")
    
    # Step 5: Train models
    print("\n🤖 STEP 5: Training models...")
    
    models_dict = {}
    histories = {}
    
    # Model 1: Efficient Transformer
    model1, history1 = train_model(
        build_efficient_transformer,
        X_train_final, y_train_final,
        X_val, y_val,
        "Efficient Transformer"
    )
    models_dict['Transformer'] = model1
    histories['Transformer'] = history1
    
    # Model 2: Hybrid CNN
    model2, history2 = train_model(
        build_hybrid_cnn,
        X_train_final, y_train_final,
        X_val, y_val,
        "Hybrid CNN"
    )
    models_dict['HybridCNN'] = model2
    histories['HybridCNN'] = history2
    
    # Model 3: Simple ResNet
    model3, history3 = train_model(
        build_resnet_simple,
        X_train_final, y_train_final,
        X_val, y_val,
        "Simple ResNet"
    )
    models_dict['ResNet'] = model3
    histories['ResNet'] = history3
    
    # Model 4: XGBoost
    model4 = train_xgboost_on_features(X_train_final, y_train_final, X_val, y_val)
    models_dict['XGBoost'] = model4
    
    # Step 6: Generate predictions
    print("\n🎯 STEP 6: Generating predictions...")
    
    predictions = {}
    
    # Deep learning models
    for name, model in models_dict.items():
        if name != 'XGBoost':
            preds = model.predict(X_val, verbose=0).flatten()
            predictions[name] = preds
            auc = roc_auc_score(y_val, preds)
            acc = accuracy_score(y_val, (preds > 0.5).astype(int))
            print(f"  {name}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")
    
    # XGBoost predictions
    def extract_simple_features(X):
        features = []
        for signal in X:
            signal_features = []
            for lead in range(12):
                lead_signal = signal[:, lead]
                signal_features.extend([
                    np.mean(lead_signal),
                    np.std(lead_signal),
                    np.max(lead_signal),
                    np.min(lead_signal),
                    np.median(lead_signal),
                    np.percentile(lead_signal, 25),
                    np.percentile(lead_signal, 75)
                ])
            features.append(signal_features)
        return np.array(features)
    
    X_val_feats = extract_simple_features(X_val)
    xgb_preds = models_dict['XGBoost'].predict_proba(X_val_feats)[:, 1]
    predictions['XGBoost'] = xgb_preds
    xgb_auc = roc_auc_score(y_val, xgb_preds)
    xgb_acc = accuracy_score(y_val, (xgb_preds > 0.5).astype(int))
    print(f"  XGBoost: AUC = {xgb_auc:.4f}, Accuracy = {xgb_acc:.4f}")
    
    # Step 7: Ensemble
    print("\n🤝 STEP 7: Creating ensemble...")
    
    # Weighted average ensemble
    ensemble_weights = {
        'Transformer': 0.3,
        'HybridCNN': 0.25,
        'ResNet': 0.25,
        'XGBoost': 0.2
    }
    
    ensemble_preds = np.zeros_like(predictions['Transformer'])
    for name, weight in ensemble_weights.items():
        ensemble_preds += predictions[name] * weight
    
    ensemble_auc = roc_auc_score(y_val, ensemble_preds)
    ensemble_acc = accuracy_score(y_val, (ensemble_preds > 0.5).astype(int))
    
    print(f"\n🎯 Ensemble Results on Validation Set:")
    print(f"  AUC:      {ensemble_auc:.4f}")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    
    # Step 8: Evaluate on test set
    print("\n🧪 STEP 8: Evaluating on test set...")
    
    # Get predictions on test set
    test_predictions = {}
    
    # Deep learning models
    for name, model in models_dict.items():
        if name != 'XGBoost':
            preds = model.predict(X_test, verbose=0).flatten()
            test_predictions[name] = preds
    
    # XGBoost on test set
    X_test_feats = extract_simple_features(X_test)
    xgb_test_preds = models_dict['XGBoost'].predict_proba(X_test_feats)[:, 1]
    test_predictions['XGBoost'] = xgb_test_preds
    
    # Ensemble on test set
    test_ensemble = np.zeros_like(test_predictions['Transformer'])
    for name, weight in ensemble_weights.items():
        test_ensemble += test_predictions[name] * weight
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, test_ensemble)
    test_acc = accuracy_score(y_test, (test_ensemble > 0.5).astype(int))
    test_cm = confusion_matrix(y_test, (test_ensemble > 0.5).astype(int))
    
    # Individual model results on test set
    print("\n📊 Individual Model Results on Test Set:")
    print("-" * 50)
    for name, preds in test_predictions.items():
        auc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        print(f"{name:<15} | AUC: {auc:.4f} | Accuracy: {acc:.4f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 FINAL ENSEMBLE RESULTS")
    print("=" * 60)
    print(f"AUC:      {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    tn, fp, fn, tp = test_cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print("\n📊 Detailed Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:         {specificity:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    
    print("\n📊 Confusion Matrix:")
    print(f"[[{tn:4d}  {fp:4d}]")
    print(f" [{fn:4d}  {tp:4d}]]")
    
    # Achievement check
    print("\n✅ TARGET ACHIEVEMENT:")
    print("-" * 50)
    if test_acc >= 0.96:
        print(f"🎉 ACCURACY TARGET ACHIEVED: {test_acc:.4f} ≥ 0.96")
    else:
        print(f"⚠️  Accuracy target not met: {test_acc:.4f} < 0.96")
    
    if test_auc >= 0.96:
        print(f"🎉 AUC TARGET ACHIEVED: {test_auc:.4f} ≥ 0.96")
    else:
        print(f"⚠️  AUC target not met: {test_auc:.4f} < 0.96")
    
    # Save results
    print("\n💾 Saving results...")
    with open(f"{CONFIG['OUTPUT_DIR']}/results.txt", "w") as f:
        f.write(f"ECG Classification Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"[[{tn} {fp}]\n")
        f.write(f" [{fn} {tp}]]\n")
        f.write(f"\nDetailed Metrics:\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    
    print(f"\n✅ Results saved to {CONFIG['OUTPUT_DIR']}/results.txt")
    print("\n🎉 Pipeline completed successfully!")

# ==============================================================================
# 6. RUN PIPELINE
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        ECG FAST PIPELINE - Optimized for Speed          ║
    ║                Target: ≥96% Accuracy & AUC              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()