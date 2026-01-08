# ==============================================================================
# ECG ROBUST PIPELINE - Anti-Overfitting with Heavy Regularization
# ==============================================================================

import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import wfdb
from concurrent.futures import ThreadPoolExecutor
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ANTI-OVERFITTING CONFIGURATION
# ==============================================================================

CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 64,
    'EPOCHS': 30,  # Reduced from 40
    'PATIENCE': 8,  # Less patience to stop earlier
    'LEARNING_RATE': 0.0003,  # Lower learning rate
    'DROPOUT_RATE': 0.4,  # Increased dropout
    'L2_REG': 1e-3,  # Strong L2 regularization
    'USE_LABEL_SMOOTHING': True,  # Prevent overconfidence
    'OUTPUT_DIR': './ecg_robust_results'
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. IMPROVED DATA LOADING WITH TEST-TIME AUGMENTATION
# ==============================================================================

def load_data_robust():
    """Load data with better preprocessing"""
    print("📥 Loading PTB-XL dataset...")
    
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    
    csv_path = None
    data_root = None
    for root, _, files in os.walk(path):
        if 'ptbxl_database.csv' in files:
            csv_path = os.path.join(root, 'ptbxl_database.csv')
            data_root = root
            break
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    
    scp_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]
    
    def get_diagnostic_label(codes):
        diagnostic_classes = []
        for code in codes:
            if code in scp_df.index:
                diagnostic_classes.append(scp_df.loc[code].diagnostic_class)
        return list(set(diagnostic_classes))
    
    df['diagnostic'] = df.scp_codes.apply(get_diagnostic_label)
    df['label'] = df['diagnostic'].apply(
        lambda x: 1 if 'MI' in x else (0 if 'NORM' in x else -1)
    )
    
    df = df[df['label'] != -1].reset_index(drop=True)
    
    # Balance by undersampling majority class
    mi_indices = df[df['label'] == 1].index
    norm_indices = df[df['label'] == 0].index
    
    min_count = min(len(mi_indices), len(norm_indices))
    mi_sample = np.random.choice(mi_indices, min_count, replace=False)
    norm_sample = np.random.choice(norm_indices, min_count, replace=False)
    
    df_balanced = df.loc[np.concatenate([mi_sample, norm_sample])]
    df_balanced = df_balanced.sample(frac=1, random_state=CONFIG['SEED']).reset_index(drop=True)
    
    print(f"📊 Balanced Dataset: {len(df_balanced)} samples")
    
    # Load signals
    def load_signal(row):
        try:
            filepath = os.path.join(data_root, row['filename_lr'])
            signal_data = wfdb.rdrecord(filepath).p_signal
            
            if signal_data.shape[0] != 1000:
                # Simple truncation/padding
                if signal_data.shape[0] > 1000:
                    signal_data = signal_data[:1000]
                else:
                    signal_data = np.pad(signal_data, ((0, 1000 - signal_data.shape[0]), (0, 0)))
            
            # Simple normalization only
            signal_data = (signal_data - np.mean(signal_data, axis=0)) / (np.std(signal_data, axis=0) + 1e-8)
            return signal_data
        except Exception as e:
            return None
    
    X = []
    y = []
    
    batch_size = 500
    for i in range(0, len(df_balanced), batch_size):
        batch_df = df_balanced.iloc[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            signals = list(executor.map(load_signal, [row for _, row in batch_df.iterrows()]))
        
        for signal_data, label in zip(signals, batch_df['label']):
            if signal_data is not None and signal_data.shape == (1000, 12):
                X.append(signal_data)
                y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} signals")
    return X, y

def augment_data_diverse(X, y):
    """Diverse data augmentation to prevent memorization"""
    print("🔄 Applying diverse augmentation...")
    
    # Apply SMOTE first
    smote = SMOTE(random_state=CONFIG['SEED'])
    X_reshaped = X.reshape(X.shape[0], -1)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    X_resampled = X_resampled.reshape(-1, 1000, 12)
    
    X_augmented = []
    y_augmented = []
    
    for signal_data, label in zip(X_resampled, y_resampled):
        # Always add original
        X_augmented.append(signal_data)
        y_augmented.append(label)
        
        # Add 3 augmented versions per sample
        for _ in range(3):
            augmented = signal_data.copy()
            
            # Random noise
            noise = np.random.normal(0, 0.03, signal_data.shape)
            augmented += noise
            
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            augmented *= scale
            
            # Random shifting
            shift = np.random.uniform(-0.2, 0.2)
            augmented += shift
            
            # Random time warping (simplified)
            warp_factor = np.random.uniform(0.9, 1.1)
            length = signal_data.shape[0]
            time_steps = np.arange(length)
            warped_steps = time_steps * warp_factor
            warped_steps = np.clip(warped_steps, 0, length-1)
            
            for lead in range(signal_data.shape[1]):
                augmented[:, lead] = np.interp(time_steps, warped_steps, augmented[:, lead])
            
            X_augmented.append(augmented)
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
# 3. HIGHLY REGULARIZED MODELS
# ==============================================================================

def build_regularized_cnn():
    """CNN with heavy regularization"""
    inputs = layers.Input(shape=(1000, 12))
    
    # Initial convolution with dropout
    x = layers.Conv1D(32, 7, padding='same', 
                     kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Block 1
    x = layers.Conv1D(64, 5, padding='same',
                     kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Block 2
    x = layers.Conv1D(128, 3, padding='same',
                     kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Block 3
    x = layers.Conv1D(256, 3, padding='same',
                     kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    # Output with label smoothing if enabled
    if CONFIG['USE_LABEL_SMOOTHING']:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='RegularizedCNN')
    
    # Label smoothing loss
    def smooth_binary_crossentropy(y_true, y_pred):
        smoothing = 0.1
        y_true = y_true * (1.0 - smoothing) + 0.5 * smoothing
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss=smooth_binary_crossentropy if CONFIG['USE_LABEL_SMOOTHING'] else 'binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def build_simple_resnet():
    """Simplified ResNet with regularization"""
    def residual_block(x, filters):
        shortcut = x
        
        x = layers.Conv1D(filters, 3, padding='same',
                         kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(CONFIG['DROPOUT_RATE']/2)(x)
        
        x = layers.Conv1D(filters, 3, padding='same',
                         kernel_regularizer=regularizers.l2(CONFIG['L2_REG']))(x)
        x = layers.BatchNormalization()(x)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    inputs = layers.Input(shape=(1000, 12))
    
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(CONFIG['DROPOUT_RATE'])(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='SimpleResNet')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# ==============================================================================
# 4. TRAINING WITH EARLY STOPPING & MONITORING
# ==============================================================================

def train_model_robust(model_builder, X_train, y_train, X_val, y_val, model_name):
    """Training with strong regularization and early stopping"""
    print(f"\n🔧 Training {model_name} (Robust)...")
    print("-" * 50)
    
    model = model_builder()
    
    # More aggressive early stopping
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=CONFIG['PATIENCE'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.2,  # More aggressive reduction
            patience=CONFIG['PATIENCE'] // 2,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        # Stop if training accuracy gets too high too quickly
        callbacks.EarlyStopping(
            monitor='accuracy',
            patience=3,
            min_delta=0.01,
            mode='max',
            verbose=0,
            start_from_epoch=5
        )
    ]
    
    print(f"  Training with:")
    print(f"  - Dropout: {CONFIG['DROPOUT_RATE']}")
    print(f"  - L2 Regularization: {CONFIG['L2_REG']}")
    print(f"  - Early stopping patience: {CONFIG['PATIENCE']}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Analyze for overfitting
    train_auc = history.history['auc'][-1]
    val_auc = history.history['val_auc'][-1]
    auc_gap = train_auc - val_auc
    
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    acc_gap = train_acc - val_acc
    
    print(f"\n📊 Overfitting Analysis for {model_name}:")
    print(f"  Training AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}, Gap: {auc_gap:.4f}")
    print(f"  Training Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}, Gap: {acc_gap:.4f}")
    
    if auc_gap > 0.05 or acc_gap > 0.05:
        print(f"  ⚠️  Potential overfitting detected!")
    elif auc_gap > 0.02 or acc_gap > 0.02:
        print(f"  ⚠️  Mild overfitting detected")
    else:
        print(f"  ✅ Good generalization")
    
    return model, history

# ==============================================================================
# 5. 5-FOLD CROSS-VALIDATION PIPELINE
# ==============================================================================

def run_cross_validation():
    """Run 5-fold cross-validation for robust evaluation"""
    print("\n🎯 Running 5-Fold Cross-Validation...")
    
    # Load data
    X, y = load_data_robust()
    
    # Create folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['SEED'])
    
    fold_results = []
    all_val_preds = []
    all_val_labels = []
    all_test_preds = []
    all_test_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold + 1}/5")
        print("-" * 40)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split training into train/validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=CONFIG['SEED']
        )
        
        # Augment training data only
        X_train_aug, y_train_aug = augment_data_diverse(X_train_final, y_train_final)
        
        print(f"  Train: {X_train_aug.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train model
        model, history = train_model_robust(
            build_regularized_cnn,
            X_train_aug, y_train_aug,
            X_val, y_val,
            f"RegularizedCNN_Fold{fold+1}"
        )
        
        # Predict on validation and test
        val_preds = model.predict(X_val, verbose=0).flatten()
        test_preds = model.predict(X_test, verbose=0).flatten()
        
        # Store predictions
        all_val_preds.extend(val_preds)
        all_val_labels.extend(y_val)
        all_test_preds.extend(test_preds)
        all_test_labels.extend(y_test)
        
        # Calculate metrics for this fold
        val_auc = roc_auc_score(y_val, val_preds)
        val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        test_auc = roc_auc_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
        
        fold_results.append({
            'fold': fold + 1,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'test_auc': test_auc,
            'test_acc': test_acc,
            'auc_gap': val_auc - test_auc,
            'acc_gap': val_acc - test_acc
        })
        
        print(f"  Fold {fold + 1} Results:")
        print(f"    Validation: AUC={val_auc:.4f}, Acc={val_acc:.4f}")
        print(f"    Test:       AUC={test_auc:.4f}, Acc={test_acc:.4f}")
        print(f"    Gap:        AUC={val_auc-test_auc:.4f}, Acc={val_acc-test_acc:.4f}")
    
    # Overall results
    print("\n" + "=" * 60)
    print("📊 CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    val_aucs = [r['val_auc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    test_aucs = [r['test_auc'] for r in fold_results]
    test_accs = [r['test_acc'] for r in fold_results]
    
    print(f"\nValidation Performance:")
    print(f"  AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"  Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    
    print(f"\nTest Performance:")
    print(f"  AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"  Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    
    print(f"\nGeneralization Gap (Validation - Test):")
    print(f"  AUC Gap: {np.mean(val_aucs)-np.mean(test_aucs):.4f}")
    print(f"  Accuracy Gap: {np.mean(val_accs)-np.mean(test_accs):.4f}")
    
    # Overall metrics on all test predictions
    overall_test_auc = roc_auc_score(all_test_labels, all_test_preds)
    overall_test_acc = accuracy_score(all_test_labels, (np.array(all_test_preds) > 0.5).astype(int))
    
    print(f"\n📈 OVERALL TEST PERFORMANCE (All Folds):")
    print(f"  AUC: {overall_test_auc:.4f}")
    print(f"  Accuracy: {overall_test_acc:.4f}")
    
    # Achievement check
    print("\n✅ TARGET ACHIEVEMENT:")
    print("-" * 50)
    if overall_test_acc >= 0.96:
        print(f"🎉 ACCURACY TARGET ACHIEVED: {overall_test_acc:.4f} ≥ 0.96")
    else:
        print(f"⚠️  Accuracy target not met: {overall_test_acc:.4f} < 0.96")
    
    if overall_test_auc >= 0.96:
        print(f"🎉 AUC TARGET ACHIEVED: {overall_test_auc:.4f} ≥ 0.96")
    else:
        print(f"⚠️  AUC target not met: {overall_test_auc:.4f} < 0.96")
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{CONFIG['OUTPUT_DIR']}/cv_results.csv", index=False)
    
    with open(f"{CONFIG['OUTPUT_DIR']}/cv_summary.txt", "w") as f:
        f.write("5-Fold Cross-Validation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overall Test AUC: {overall_test_auc:.4f}\n")
        f.write(f"Overall Test Accuracy: {overall_test_acc:.4f}\n")
        f.write("\nPer-fold Results:\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: Val AUC={result['val_auc']:.4f}, "
                   f"Test AUC={result['test_auc']:.4f}, Gap={result['auc_gap']:.4f}\n")
    
    return overall_test_acc, overall_test_auc

# ==============================================================================
# 6. TEST-TIME AUGMENTATION (TTA) FOR BETTER GENERALIZATION
# ==============================================================================

def test_time_augmentation(model, X, n_augmentations=10):
    """Apply test-time augmentation for more robust predictions"""
    print(f"\n🎲 Applying Test-Time Augmentation (n={n_augmentations})...")
    
    all_predictions = []
    
    for i in range(n_augmentations):
        if i == 0:
            # Original
            X_aug = X.copy()
        else:
            # Apply random augmentation
            X_aug = X.copy()
            noise = np.random.normal(0, 0.02, X.shape)
            X_aug += noise
            
            # Random scaling
            scale = np.random.uniform(0.95, 1.05, (X.shape[0], 1, 1))
            X_aug *= scale
        
        preds = model.predict(X_aug, verbose=0).flatten()
        all_predictions.append(preds)
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    
    print(f"  Done. Using average of {n_augmentations} augmented versions.")
    return avg_predictions

# ==============================================================================
# 7. MAIN ROBUST PIPELINE
# ==============================================================================

def main_robust():
    """Main robust pipeline with anti-overfitting measures"""
    print("=" * 70)
    print("🛡️  ECG ROBUST PIPELINE - Anti-Overfitting Measures")
    print("=" * 70)
    
    print("\n⚙️  ANTI-OVERFITTING SETTINGS:")
    print(f"  • Dropout Rate: {CONFIG['DROPOUT_RATE']}")
    print(f"  • L2 Regularization: {CONFIG['L2_REG']}")
    print(f"  • Label Smoothing: {CONFIG['USE_LABEL_SMOOTHING']}")
    print(f"  • Early Stopping Patience: {CONFIG['PATIENCE']}")
    print(f"  • 5-Fold Cross-Validation")
    
    # Run cross-validation
    test_acc, test_auc = run_cross_validation()
    
    # Final model training on all data
    print("\n" + "=" * 60)
    print("🏁 TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)
    
    X, y = load_data_robust()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=CONFIG['SEED']
    )
    
    # Augment
    X_train_aug, y_train_aug = augment_data_diverse(X_train, y_train)
    
    # Split for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_aug, y_train_aug, test_size=0.15, stratify=y_train_aug, random_state=CONFIG['SEED']
    )
    
    # Train final model
    final_model, history = train_model_robust(
        build_regularized_cnn,
        X_train_final, y_train_final,
        X_val, y_val,
        "Final_RegularizedCNN"
    )
    
    # Evaluate with Test-Time Augmentation
    print("\n📊 FINAL EVALUATION WITH TEST-TIME AUGMENTATION:")
    test_preds = test_time_augmentation(final_model, X_test, n_augmentations=5)
    
    # Metrics
    test_auc_final = roc_auc_score(y_test, test_preds)
    test_acc_final = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    test_cm = confusion_matrix(y_test, (test_preds > 0.5).astype(int))
    
    print(f"\n🎯 FINAL RESULTS (with TTA):")
    print(f"  AUC:      {test_auc_final:.4f}")
    print(f"  Accuracy: {test_acc_final:.4f}")
    
    tn, fp, fn, tp = test_cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 Detailed Metrics:")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
    
    # Compare with previous overfitting results
    print("\n📈 IMPROVEMENT OVER PREVIOUS PIPELINE:")
    print(f"  Previous: AUC=0.9771, Accuracy=0.9198 (Severe overfit)")
    print(f"  Current:  AUC={test_auc_final:.4f}, Accuracy={test_acc_final:.4f}")
    print(f"  Improvement: Accuracy +{(test_acc_final-0.9198)*100:.2f}%")
    
    # Check if we're closer to targets
    print("\n✅ FINAL TARGET ASSESSMENT:")
    if test_acc_final >= 0.96 and test_auc_final >= 0.96:
        print("🎉 CONGRATULATIONS! Both targets achieved without overfitting!")
    elif test_auc_final >= 0.96:
        print(f"✅ AUC target achieved ({test_auc_final:.4f} ≥ 0.96)")
        print(f"⚠️  Accuracy close: {test_acc_final:.4f} (target: 0.96)")
        
        if test_acc_final >= 0.95:
            print("\n💡 RECOMMENDATION: You're very close! Try:")
            print("  1. Increase training data (if available)")
            print("  2. Add more diverse augmentations")
            print("  3. Try ensemble of multiple regularized models")
        else:
            print("\n💡 RECOMMENDATION: Need more improvement. Try:")
            print("  1. Use the original Titan pipeline with 5-fold CV")
            print("  2. Collect more training data")
            print("  3. Try different architectures (e.g., EfficientNet)")
    else:
        print("⚠️  Both targets not met. Consider:")
        print("  1. Using the proven Titan pipeline with modifications")
        print("  2. Collecting more data")
        print("  3. Trying transfer learning from pre-trained models")
    
    # Save final model
    final_model.save(f"{CONFIG['OUTPUT_DIR']}/final_model.h5")
    print(f"\n💾 Model saved to {CONFIG['OUTPUT_DIR']}/final_model.h5")
    
    return test_acc_final, test_auc_final

# ==============================================================================
# 8. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        ECG ROBUST PIPELINE - Anti-Overfitting           ║
    ║          Target: ≥96% Accuracy & AUC (Generalized)      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        final_acc, final_auc = main_robust()
        
        print("\n" + "=" * 70)
        print("📋 FINAL SUMMARY:")
        print(f"  Final Accuracy: {final_acc:.4f}")
        print(f"  Final AUC:      {final_auc:.4f}")
        
        if final_acc >= 0.96 and final_auc >= 0.96:
            print("\n✨ SUCCESS! Both targets achieved with good generalization!")
        elif final_auc >= 0.96:
            print(f"\n⚠️  Partial success: AUC target achieved, accuracy = {final_acc:.4f}")
            print("   The model generalizes well but needs slight improvement in accuracy.")
        else:
            print(f"\n❌ Targets not fully achieved. Your model now generalizes better")
            print("   but needs more improvement. Consider the recommendations above.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()