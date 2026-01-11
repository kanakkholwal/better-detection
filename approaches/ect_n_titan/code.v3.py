# ==============================================================================
# ECG CLINICAL PIPELINE - Clinically Usable with High Performance
# ==============================================================================
# Target: ≥96% Accuracy & AUC for Clinical Deployment
# Features: Clinical validation, interpretability, and real-world robustness

import ast
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CLINICAL CONFIGURATION
# ==============================================================================

CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 32,  # Smaller for better gradient updates
    'EPOCHS': 50,  # More epochs for better convergence
    'PATIENCE': 10,
    'LEARNING_RATE': 0.001,
    'DROPOUT_RATE': 0.25,  # Moderate dropout
    'L2_REG': 1e-4,  # Mild regularization
    'USE_GRADIENT_CLIPPING': True,
    'CLIP_VALUE': 1.0,
    'ENSEMBLE_SIZE': 3,  # Number of models in ensemble
    'CROSS_VALIDATION_FOLDS': 5,
    'OUTPUT_DIR': './ecg_clinical_results',
    'MIN_SENSITIVITY': 0.95,  # Clinical requirement for MI detection
    'MIN_SPECIFICITY': 0.95,
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ==============================================================================
# 2. CLINICAL DATA LOADING WITH QUALITY CONTROL
# ==============================================================================

class ClinicalECGDataLoader:
    """Clinical-grade ECG data loading with signal quality assessment"""
    
    def __init__(self):
        self.quality_threshold = 0.8  # Minimum signal quality score
        
    def load_dataset(self):
        """Load PTB-XL dataset with clinical validation"""
        print("📥 Loading PTB-XL dataset for clinical validation...")
        
        # Download dataset
        path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
        
        # Find files
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
        
        # Clinical labeling: MI vs Normal with confidence scores
        def get_clinical_label(codes):
            mi_confidence = 0
            norm_confidence = 0
            
            for code in codes:
                if code in scp_df.index:
                    diag_class = scp_df.loc[code].diagnostic_class
                    if diag_class == 'MI':
                        mi_confidence += 1
                    elif diag_class == 'NORM':
                        norm_confidence += 1
            
            # Return label with confidence
            if mi_confidence > norm_confidence:
                return 1, mi_confidence / (mi_confidence + norm_confidence)
            elif norm_confidence > mi_confidence:
                return 0, norm_confidence / (mi_confidence + norm_confidence)
            else:
                return -1, 0.5
        
        labels = []
        confidences = []
        for codes in df.scp_codes:
            label, confidence = get_clinical_label(codes)
            labels.append(label)
            confidences.append(confidence)
        
        df['label'] = labels
        df['confidence'] = confidences
        
        # Filter only high-confidence samples
        df = df[(df['label'] != -1) & (df['confidence'] > 0.7)].reset_index(drop=True)
        
        print(f"📊 Clinical Dataset: {len(df)} high-confidence samples")
        print(f"   MI (Myocardial Infarction): {sum(df['label'] == 1)}")
        print(f"   Normal: {sum(df['label'] == 0)}")
        
        return df, data_root
    
    def load_signal_with_quality(self, row, data_root):
        """Load ECG signal with quality assessment"""
        try:
            filepath = os.path.join(data_root, row['filename_lr'])
            record = wfdb.rdrecord(filepath)
            signal_data = record.p_signal
            
            # Check signal quality
            quality_score = self.assess_signal_quality(signal_data)
            
            if quality_score < self.quality_threshold:
                print(f"⚠️  Low quality signal: {quality_score:.2f}")
                return None, quality_score
            
            # Ensure correct shape (1000 samples, 12 leads)
            if signal_data.shape[0] != 1000:
                signal_data = self.resample_to_1000(signal_data)
            
            # Clinical preprocessing
            signal_data = self.clinical_preprocessing(signal_data)
            
            return signal_data, quality_score
            
        except Exception as e:
            print(f"❌ Error loading signal: {e}")
            return None, 0.0
    
    def assess_signal_quality(self, signal):
        """Assess ECG signal quality (simplified version)"""
        quality_metrics = []
        
        for lead in range(signal.shape[1]):
            lead_signal = signal[:, lead]
            
            # Check for flat line
            if np.std(lead_signal) < 0.01:
                quality_metrics.append(0.0)
                continue
            
            # Check for excessive noise
            noise_level = np.std(np.diff(lead_signal))
            if noise_level > 5.0:
                quality_metrics.append(0.3)
                continue
            
            # Check for saturation
            if np.max(np.abs(lead_signal)) > 10.0:
                quality_metrics.append(0.5)
                continue
            
            # Good signal
            quality_metrics.append(1.0)
        
        return np.mean(quality_metrics)
    
    def resample_to_1000(self, signal):
        """Resample signal to 1000 samples"""
        if signal.shape[0] > 1000:
            # Simple decimation
            step = signal.shape[0] // 1000
            return signal[::step][:1000]
        else:
            # Linear interpolation
            from scipy import interpolate
            x_old = np.linspace(0, 1, signal.shape[0])
            x_new = np.linspace(0, 1, 1000)
            f = interpolate.interp1d(x_old, signal, axis=0, kind='linear')
            return f(x_new)
    
    def clinical_preprocessing(self, signal):
        """Clinical-grade ECG preprocessing"""
        processed = signal.copy()
        
        # 1. Baseline wander removal (simplified)
        for lead in range(signal.shape[1]):
            lead_signal = signal[:, lead]
            baseline = np.convolve(lead_signal, np.ones(50)/50, mode='same')
            processed[:, lead] = lead_signal - baseline
        
        # 2. Powerline interference removal (50/60 Hz)
        # Simplified: notch filter approximation
        fs = 500  # Sampling frequency
        for lead in range(signal.shape[1]):
            fft_signal = np.fft.fft(processed[:, lead])
            freqs = np.fft.fftfreq(len(fft_signal), 1/fs)
            
            # Remove 50Hz and 60Hz components
            for freq in [50, 60]:
                idx = np.where(np.abs(freqs - freq) < 2)[0]
                fft_signal[idx] = 0
            
            processed[:, lead] = np.real(np.fft.ifft(fft_signal))
        
        # 3. Standardization per lead
        for lead in range(signal.shape[1]):
            lead_signal = processed[:, lead]
            if np.std(lead_signal) > 0:
                processed[:, lead] = (lead_signal - np.mean(lead_signal)) / np.std(lead_signal)
        
        return processed

# ==============================================================================
# 3. CLINICAL DATA AUGMENTATION
# ==============================================================================

class ClinicalAugmentor:
    """Clinical-realistic data augmentation"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def augment_batch(self, signals, labels, augmentation_factor=2):
        """Apply clinically-realistic augmentations"""
        augmented_signals = []
        augmented_labels = []
        
        for signal, label in zip(signals, labels):
            # Always keep original
            augmented_signals.append(signal)
            augmented_labels.append(label)
            
            # Generate augmented versions
            for _ in range(augmentation_factor):
                aug_signal = signal.copy()
                
                # Clinical augmentations
                aug_signal = self.apply_random_noise(aug_signal)
                aug_signal = self.apply_baseline_wander(aug_signal)
                aug_signal = self.apply_lead_swap(aug_signal)
                aug_signal = self.apply_time_warp(aug_signal)
                aug_signal = self.apply_amplitude_variation(aug_signal)
                
                augmented_signals.append(aug_signal)
                augmented_labels.append(label)
        
        return np.array(augmented_signals), np.array(augmented_labels)
    
    def apply_random_noise(self, signal, noise_level=0.02):
        """Add random noise (simulating electrode contact issues)"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    def apply_baseline_wander(self, signal, wander_freq=0.5, amplitude=0.1):
        """Simulate baseline wander"""
        t = np.arange(signal.shape[0])
        wander = amplitude * np.sin(2 * np.pi * wander_freq * t / signal.shape[0])
        wander = wander[:, np.newaxis]  # Add channel dimension
        return signal + wander
    
    def apply_lead_swap(self, signal):
        """Randomly swap leads (simulating misplacement)"""
        if np.random.random() < 0.1:  # 10% chance
            # Swap random pair of leads
            lead1, lead2 = np.random.choice(12, 2, replace=False)
            signal[:, [lead1, lead2]] = signal[:, [lead2, lead1]]
        return signal
    
    def apply_time_warp(self, signal, warp_factor=0.9):
        """Apply time warping (simulating heart rate variability)"""
        if np.random.random() < 0.3:  # 30% chance
            from scipy import interpolate
            original_length = signal.shape[0]
            warp = np.random.uniform(warp_factor, 1/warp_factor)
            new_length = int(original_length * warp)
            
            warped_signal = np.zeros((new_length, signal.shape[1]))
            for lead in range(signal.shape[1]):
                f = interpolate.interp1d(
                    np.linspace(0, 1, original_length),
                    signal[:, lead],
                    kind='cubic'
                )
                warped_signal[:, lead] = f(np.linspace(0, 1, new_length))
            
            # Resample back to original length
            if new_length > original_length:
                step = new_length // original_length
                signal = warped_signal[::step][:original_length]
            else:
                f = interpolate.interp1d(
                    np.linspace(0, 1, new_length),
                    warped_signal,
                    axis=0,
                    kind='cubic'
                )
                signal = f(np.linspace(0, 1, original_length))
        
        return signal
    
    def apply_amplitude_variation(self, signal, variation=0.2):
        """Apply amplitude variation (simulating gain changes)"""
        scale = np.random.uniform(1 - variation, 1 + variation)
        return signal * scale

# ==============================================================================
# 4. CLINICAL MODELS
# ==============================================================================

class ClinicalECGModels:
    """Clinically-optimized ECG models"""
    
    @staticmethod
    def build_clinical_cnn(input_shape=(1000, 12)):
        """CNN optimized for clinical ECG analysis"""
        inputs = layers.Input(shape=input_shape)
        
        # Lead-wise initial processing
        x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Multi-scale feature extraction
        # Branch 1: High-frequency features
        branch1 = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Conv1D(64, 3, padding='same', activation='relu')(branch1)
        branch1 = layers.BatchNormalization()(branch1)
        
        # Branch 2: Low-frequency features
        branch2 = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
        branch2 = layers.BatchNormalization()(branch2)
        
        # Combine branches
        x = layers.Concatenate()([branch1, branch2])
        x = layers.MaxPooling1D(2)(x)
        
        # Deep feature extraction
        x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Clinical features layer
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output with clinical calibration
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ClinicalCNN')
        
        # Optimizer with gradient clipping
        optimizer = optimizers.Adam(
            learning_rate=CONFIG['LEARNING_RATE'],
            clipvalue=CONFIG['CLIP_VALUE'] if CONFIG['USE_GRADIENT_CLIPPING'] else None
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    @staticmethod
    def build_resnet_clinical(input_shape=(1000, 12)):
        """ResNet adapted for clinical ECG"""
        
        def residual_block(x, filters, kernel_size=3, stride=1):
            shortcut = x
            
            # Main path
            x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv1D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Shortcut connection
            if shortcut.shape[-1] != filters or stride != 1:
                shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            return x
        
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ClinicalResNet')
        
        optimizer = optimizers.Adam(
            learning_rate=CONFIG['LEARNING_RATE'],
            clipvalue=CONFIG['CLIP_VALUE'] if CONFIG['USE_GRADIENT_CLIPPING'] else None
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    @staticmethod
    def build_attention_model(input_shape=(1000, 12)):
        """Attention-based model for interpretability"""
        inputs = layers.Input(shape=input_shape)
        
        # Feature extraction
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Attention mechanism
        attention = layers.Conv1D(1, 1, activation='sigmoid')(x)
        attention = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling1D()(attention)
        
        # Classification
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='AttentionModel')
        
        optimizer = optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

# ==============================================================================
# 5. CLINICAL TRAINING PIPELINE
# ==============================================================================

class ClinicalTrainer:
    """Clinical-grade training pipeline"""
    
    def __init__(self):
        self.data_loader = ClinicalECGDataLoader()
        self.augmentor = ClinicalAugmentor(seed=CONFIG['SEED'])
        self.models = ClinicalECGModels()
    
    def load_and_prepare_data(self):
        """Load and prepare clinical data"""
        print("🩺 Loading clinical ECG data...")
        
        df, data_root = self.data_loader.load_dataset()
        
        # Load signals with quality control
        X = []
        y = []
        quality_scores = []
        
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_signals = []
            batch_labels = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _, row in batch_df.iterrows():
                    futures.append(
                        executor.submit(
                            self.data_loader.load_signal_with_quality,
                            row, data_root
                        )
                    )
                
                for future, label in zip(futures, batch_df['label']):
                    signal, quality = future.result()
                    if signal is not None:
                        batch_signals.append(signal)
                        batch_labels.append(label)
                        quality_scores.append(quality)
            
            X.extend(batch_signals)
            y.extend(batch_labels)
            
            print(f"  Loaded {len(X)}/{len(df)} high-quality signals...")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"✅ Final dataset: {len(X)} high-quality signals")
        print(f"   Average quality score: {np.mean(quality_scores):.2f}")
        
        return X, y
    
    def train_with_cross_validation(self):
        """Train with 5-fold cross-validation for clinical validation"""
        print("\n🎯 Starting 5-fold cross-validation for clinical validation...")
        
        X, y = self.load_and_prepare_data()
        
        skf = StratifiedKFold(
            n_splits=CONFIG['CROSS_VALIDATION_FOLDS'],
            shuffle=True,
            random_state=CONFIG['SEED']
        )
        
        fold_results = []
        all_test_preds = []
        all_test_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n🔁 Fold {fold + 1}/{CONFIG['CROSS_VALIDATION_FOLDS']}")
            print("-" * 40)
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Split train into train/val
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train,
                test_size=0.15,
                stratify=y_train,
                random_state=CONFIG['SEED']
            )
            
            # Apply clinical augmentation
            print("  Applying clinical augmentation...")
            X_train_aug, y_train_aug = self.augmentor.augment_batch(
                X_train_final, y_train_final, augmentation_factor=2
            )
            
            print(f"  Training samples: {len(X_train_aug)}")
            print(f"  Validation samples: {len(X_val)}")
            print(f"  Test samples: {len(X_test)}")
            
            # Train ensemble of models
            models = []
            histories = []
            
            # Train Clinical CNN
            print("\n  Training Clinical CNN...")
            model1 = self.models.build_clinical_cnn()
            history1 = self.train_model(
                model1, X_train_aug, y_train_aug, X_val, y_val,
                f"ClinicalCNN_Fold{fold+1}"
            )
            models.append(model1)
            histories.append(history1)
            
            # Train Clinical ResNet
            print("\n  Training Clinical ResNet...")
            model2 = self.models.build_resnet_clinical()
            history2 = self.train_model(
                model2, X_train_aug, y_train_aug, X_val, y_val,
                f"ClinicalResNet_Fold{fold+1}"
            )
            models.append(model2)
            histories.append(history2)
            
            # Train Attention Model
            print("\n  Training Attention Model...")
            model3 = self.models.build_attention_model()
            history3 = self.train_model(
                model3, X_train_aug, y_train_aug, X_val, y_val,
                f"AttentionModel_Fold{fold+1}"
            )
            models.append(model3)
            histories.append(history3)
            
            # Create ensemble predictions
            print("\n  Creating ensemble predictions...")
            val_preds = []
            test_preds = []
            
            for model in models:
                val_pred = model.predict(X_val, verbose=0).flatten()
                test_pred = model.predict(X_test, verbose=0).flatten()
                val_preds.append(val_pred)
                test_preds.append(test_pred)
            
            # Weighted ensemble based on validation AUC
            val_aucs = []
            for pred in val_preds:
                auc_score = roc_auc_score(y_val, pred)
                val_aucs.append(auc_score)
            
            # Softmax weights
            val_aucs = np.array(val_aucs)
            weights = np.exp(val_aucs) / np.sum(np.exp(val_aucs))
            
            # Ensemble predictions
            ensemble_val_pred = np.zeros_like(val_preds[0])
            ensemble_test_pred = np.zeros_like(test_preds[0])
            
            for i, weight in enumerate(weights):
                ensemble_val_pred += val_preds[i] * weight
                ensemble_test_pred += test_preds[i] * weight
            
            # Calculate metrics
            val_auc = roc_auc_score(y_val, ensemble_val_pred)
            val_acc = accuracy_score(y_val, (ensemble_val_pred > 0.5).astype(int))
            
            test_auc = roc_auc_score(y_test, ensemble_test_pred)
            test_acc = accuracy_score(y_test, (ensemble_test_pred > 0.5).astype(int))
            
            # Clinical metrics
            test_pred_binary = (ensemble_test_pred > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, test_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            fold_results.append({
                'fold': fold + 1,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'test_auc': test_auc,
                'test_acc': test_acc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })
            
            # Store for overall evaluation
            all_test_preds.extend(ensemble_test_pred)
            all_test_labels.extend(y_test)
            
            print(f"\n  📊 Fold {fold + 1} Results:")
            print(f"    Validation - AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
            print(f"    Test - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")
            print(f"    Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        
        # Overall evaluation
        print("\n" + "=" * 70)
        print("🎯 CLINICAL VALIDATION RESULTS")
        print("=" * 70)
        
        self.evaluate_clinical_performance(all_test_labels, all_test_preds, fold_results)
        
        return all_test_labels, all_test_preds, fold_results
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Train a single model with clinical monitoring"""
        
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
                factor=0.5,
                patience=CONFIG['PATIENCE'] // 2,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{CONFIG['OUTPUT_DIR']}/{model_name}.h5",
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=0
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['EPOCHS'],
            batch_size=CONFIG['BATCH_SIZE'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate_clinical_performance(self, y_true, y_pred, fold_results):
        """Comprehensive clinical evaluation"""
        
        # Overall metrics
        overall_auc = roc_auc_score(y_true, y_pred)
        overall_acc = accuracy_score(y_true, (np.array(y_pred) > 0.5).astype(int))
        
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Calculate fold averages
        fold_aucs = [r['test_auc'] for r in fold_results]
        fold_accs = [r['test_acc'] for r in fold_results]
        fold_sens = [r['sensitivity'] for r in fold_results]
        fold_spec = [r['specificity'] for r in fold_results]
        
        print(f"\n📊 OVERALL CLINICAL PERFORMANCE:")
        print(f"  AUC:              {overall_auc:.4f} (mean ± std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})")
        print(f"  Accuracy:         {overall_acc:.4f} (mean ± std: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f})")
        print(f"  Sensitivity:      {sensitivity:.4f} (mean ± std: {np.mean(fold_sens):.4f} ± {np.std(fold_sens):.4f})")
        print(f"  Specificity:      {specificity:.4f} (mean ± std: {np.mean(fold_spec):.4f} ± {np.std(fold_spec):.4f})")
        print(f"  Precision:        {precision:.4f}")
        print(f"  F1-Score:         {f1:.4f}")
        print(f"  Positive Samples: {tp + fn}")
        print(f"  Negative Samples: {tn + fp}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"        Predicted")
        print(f"       0     1")
        print(f"True 0 [{tn:4d}  {fp:4d}]")
        print(f"     1 [{fn:4d}  {tp:4d}]")
        
        # Clinical achievement check
        print("\n✅ CLINICAL TARGET ASSESSMENT:")
        print("-" * 50)
        
        targets_met = []
        
        if overall_acc >= 0.96:
            print(f"🎉 ACCURACY TARGET ACHIEVED: {overall_acc:.4f} ≥ 0.96")
            targets_met.append('accuracy')
        else:
            print(f"⚠️  Accuracy: {overall_acc:.4f} < 0.96")
        
        if overall_auc >= 0.96:
            print(f"🎉 AUC TARGET ACHIEVED: {overall_auc:.4f} ≥ 0.96")
            targets_met.append('auc')
        else:
            print(f"⚠️  AUC: {overall_auc:.4f} < 0.96")
        
        if sensitivity >= CONFIG['MIN_SENSITIVITY']:
            print(f"🎉 SENSITIVITY TARGET ACHIEVED: {sensitivity:.4f} ≥ {CONFIG['MIN_SENSITIVITY']}")
            targets_met.append('sensitivity')
        else:
            print(f"⚠️  Sensitivity: {sensitivity:.4f} < {CONFIG['MIN_SENSITIVITY']}")
        
        if specificity >= CONFIG['MIN_SPECIFICITY']:
            print(f"🎉 SPECIFICITY TARGET ACHIEVED: {specificity:.4f} ≥ {CONFIG['MIN_SPECIFICITY']}")
            targets_met.append('specificity')
        else:
            print(f"⚠️  Specificity: {specificity:.4f} < {CONFIG['MIN_SPECIFICITY']}")
        
        # Recommendations
        print("\n💡 CLINICAL RECOMMENDATIONS:")
        if len(targets_met) == 4:
            print("✅ EXCELLENT! All clinical targets met. Model is ready for clinical validation studies.")
        elif 'sensitivity' in targets_met and 'specificity' in targets_met:
            print("✅ GOOD! Sensitivity and specificity targets met. Clinical utility is high.")
            if overall_acc < 0.96:
                print("   Consider: Fine-tuning on more diverse data to improve overall accuracy.")
        else:
            print("🔄 NEEDS IMPROVEMENT:")
            if sensitivity < CONFIG['MIN_SENSITIVITY']:
                print("   • Focus on improving MI detection sensitivity")
                print("   • Add more MI-positive examples to training")
                print("   • Adjust classification threshold")
            
            if specificity < CONFIG['MIN_SPECIFICITY']:
                print("   • Reduce false positives for Normal cases")
                print("   • Add more Normal examples with variations")
                print("   • Consider using rejection option for uncertain cases")
        
        # Save detailed results
        self.save_clinical_report(
            y_true, y_pred, overall_auc, overall_acc,
            sensitivity, specificity, precision, f1,
            cm, fold_results
        )
    
    def save_clinical_report(self, y_true, y_pred, auc, acc, sens, spec, prec, f1, cm, fold_results):
        """Save comprehensive clinical report"""
        
        report_path = f"{CONFIG['OUTPUT_DIR']}/clinical_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ECG MI DETECTION - CLINICAL VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"AUC Score:           {auc:.4f}\n")
            f.write(f"Accuracy:            {acc:.4f}\n")
            f.write(f"Sensitivity (Recall): {sens:.4f}\n")
            f.write(f"Specificity:         {spec:.4f}\n")
            f.write(f"Precision:           {prec:.4f}\n")
            f.write(f"F1-Score:            {f1:.4f}\n\n")
            
            f.write("CLINICAL TARGET ACHIEVEMENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy ≥ 96%:   {'✓' if acc >= 0.96 else '✗'}\n")
            f.write(f"AUC ≥ 96%:       {'✓' if auc >= 0.96 else '✗'}\n")
            f.write(f"Sensitivity ≥ 95%: {'✓' if sens >= 0.95 else '✗'}\n")
            f.write(f"Specificity ≥ 95%: {'✓' if spec >= 0.95 else '✗'}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            tn, fp, fn, tp = cm.ravel()
            f.write(f"                 Predicted\n")
            f.write(f"                 Normal    MI\n")
            f.write(f"Actual Normal    {tn:6d}    {fp:6d}\n")
            f.write(f"Actual MI        {fn:6d}    {tp:6d}\n\n")
            
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write("Fold |   AUC   | Accuracy | Sens. | Spec.\n")
            f.write("-" * 40 + "\n")
            for result in fold_results:
                f.write(f"{result['fold']:4d} | {result['test_auc']:.4f} | {result['test_acc']:.4f} "
                       f"| {result['sensitivity']:.4f} | {result['specificity']:.4f}\n")
            
            f.write("\nCLINICAL RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if acc >= 0.96 and auc >= 0.96 and sens >= 0.95 and spec >= 0.95:
                f.write("✅ Model meets all clinical requirements.\n")
                f.write("   Ready for prospective clinical validation studies.\n")
            else:
                f.write("⚠️  Model needs improvement before clinical use:\n")
                if sens < 0.95:
                    f.write("   • Improve sensitivity for MI detection\n")
                if spec < 0.95:
                    f.write("   • Improve specificity to reduce false positives\n")
                f.write("\n   Suggested actions:\n")
                f.write("   1. Collect more diverse training data\n")
                f.write("   2. Implement ensemble with different architectures\n")
                f.write("   3. Use clinical feature engineering\n")
                f.write("   4. Consider transfer learning from larger datasets\n")
        
        print(f"\n💾 Clinical report saved to: {report_path}")

# ==============================================================================
# 6. CLINICAL DEPLOYMENT PIPELINE
# ==============================================================================

class ClinicalDeployment:
    """Pipeline for clinical deployment"""
    
    def __init__(self):
        self.trainer = ClinicalTrainer()
    
    def run_full_pipeline(self):
        """Run the complete clinical pipeline"""
        print("=" * 80)
        print("🏥 ECG CLINICAL DEPLOYMENT PIPELINE")
        print("Target: ≥96% Accuracy & AUC for Myocardial Infarction Detection")
        print("=" * 80)
        
        print("\n🩺 CLINICAL REQUIREMENTS:")
        print(f"  • Minimum Accuracy: 96%")
        print(f"  • Minimum AUC: 96%")
        print(f"  • Minimum Sensitivity: 95% (for MI detection)")
        print(f"  • Minimum Specificity: 95% (to avoid unnecessary interventions)")
        
        # Run cross-validation
        y_true, y_pred, fold_results = self.trainer.train_with_cross_validation()
        
        # Train final model on all data
        print("\n" + "=" * 70)
        print("🏁 TRAINING FINAL CLINICAL MODEL")
        print("=" * 70)
        
        X, y = self.trainer.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=CONFIG['SEED']
        )
        
        # Augment training data
        X_train_aug, y_train_aug = self.trainer.augmentor.augment_batch(
            X_train, y_train, augmentation_factor=3
        )
        
        # Split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_aug, y_train_aug, test_size=0.15, stratify=y_train_aug, random_state=CONFIG['SEED']
        )
        
        # Train final ensemble
        print("\n🔧 Training final clinical ensemble...")
        
        final_models = []
        final_model_names = [
            'ClinicalCNN_Final',
            'ClinicalResNet_Final',
            'AttentionModel_Final'
        ]
        
        model_builders = [
            self.trainer.models.build_clinical_cnn,
            self.trainer.models.build_resnet_clinical,
            self.trainer.models.build_attention_model
        ]
        
        for model_name, builder in zip(final_model_names, model_builders):
            print(f"\n  Training {model_name}...")
            model = builder()
            
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=CONFIG['EPOCHS'],
                batch_size=CONFIG['BATCH_SIZE'],
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_auc',
                        patience=CONFIG['PATIENCE'],
                        restore_best_weights=True,
                        mode='max'
                    )
                ],
                verbose=0
            )
            
            final_models.append(model)
            
            # Save model
            model.save(f"{CONFIG['OUTPUT_DIR']}/{model_name}.h5")
            print(f"    Saved to {CONFIG['OUTPUT_DIR']}/{model_name}.h5")
        
        # Create final ensemble
        print("\n🤝 Creating final clinical ensemble...")
        
        # Get predictions from all models
        val_preds = []
        test_preds = []
        
        for model in final_models:
            val_pred = model.predict(X_val, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            val_preds.append(val_pred)
            test_preds.append(test_pred)
        
        # Calculate optimal weights
        val_aucs = [roc_auc_score(y_val, pred) for pred in val_preds]
        weights = np.array(val_aucs) / np.sum(val_aucs)
        
        print("\n📊 Ensemble Weights:")
        for name, weight in zip(final_model_names, weights):
            print(f"  {name}: {weight:.3f}")
        
        # Weighted ensemble predictions
        final_val_pred = np.zeros_like(val_preds[0])
        final_test_pred = np.zeros_like(test_preds[0])
        
        for i, weight in enumerate(weights):
            final_val_pred += val_preds[i] * weight
            final_test_pred += test_preds[i] * weight
        
        # Final evaluation
        final_auc = roc_auc_score(y_test, final_test_pred)
        final_acc = accuracy_score(y_test, (final_test_pred > 0.5).astype(int))
        
        y_pred_binary = (final_test_pred > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\n🎯 FINAL CLINICAL MODEL PERFORMANCE:")
        print(f"  Test AUC:        {final_auc:.4f}")
        print(f"  Test Accuracy:   {final_acc:.4f}")
        print(f"  Sensitivity:     {sensitivity:.4f}")
        print(f"  Specificity:     {specificity:.4f}")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        
        # Save final ensemble
        ensemble_info = {
            'models': final_model_names,
            'weights': weights.tolist(),
            'performance': {
                'auc': float(final_auc),
                'accuracy': float(final_acc),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity)
            }
        }
        
        import json
        with open(f"{CONFIG['OUTPUT_DIR']}/ensemble_config.json", 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print(f"\n💾 Ensemble configuration saved to: {CONFIG['OUTPUT_DIR']}/ensemble_config.json")
        
        # Clinical readiness assessment
        print("\n" + "=" * 70)
        print("🏥 CLINICAL READINESS ASSESSMENT")
        print("=" * 70)
        
        if final_acc >= 0.96 and final_auc >= 0.96 and sensitivity >= 0.95 and specificity >= 0.95:
            print("✅ CLINICALLY READY!")
            print("   The model meets all clinical requirements for MI detection.")
            print("   Next steps:")
            print("   1. Prospective validation on new patient data")
            print("   2. Regulatory approval process")
            print("   3. Integration with hospital systems")
        else:
            print("⚠️  NOT YET CLINICALLY READY")
            print("   Model needs improvement in:")
            if final_acc < 0.96:
                print(f"   • Accuracy: {final_acc:.4f} < 0.96")
            if final_auc < 0.96:
                print(f"   • AUC: {final_auc:.4f} < 0.96")
            if sensitivity < 0.95:
                print(f"   • Sensitivity: {sensitivity:.4f} < 0.95")
            if specificity < 0.95:
                print(f"   • Specificity: {specificity:.4f} < 0.95")
            
            print("\n   Recommendations:")
            print("   1. Collect more diverse training data")
            print("   2. Implement more sophisticated data augmentation")
            print("   3. Use transfer learning from larger ECG datasets")
            print("   4. Consider clinical feature engineering")
        
        return final_acc, final_auc, sensitivity, specificity

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        ECG CLINICAL DEPLOYMENT PIPELINE                 ║
    ║      Myocardial Infarction Detection System             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("This pipeline implements a clinically-focused ECG analysis system")
    print("for detecting Myocardial Infarction (Heart Attack).")
    print("\nKey Features:")
    print("  • Clinical-grade signal quality assessment")
    print("  • Realistic data augmentation")
    print("  • Ensemble of specialized models")
    print("  • 5-fold cross-validation for robust evaluation")
    print("  • Clinical metrics (Sensitivity, Specificity)")
    print("  • Deployment-ready model saving")
    
    print("\n" + "=" * 60)
    print("TARGETS FOR CLINICAL USE:")
    print("  Accuracy: ≥96%")
    print("  AUC: ≥96%")
    print("  Sensitivity: ≥95% (for MI detection)")
    print("  Specificity: ≥95% (to avoid false alarms)")
    print("=" * 60)
    
    choice = input("\nStart clinical pipeline? (y/n): ").strip().lower()
    
    if choice == 'y':
        try:
            deployment = ClinicalDeployment()
            accuracy, auc, sensitivity, specificity = deployment.run_full_pipeline()
            
            print("\n" + "=" * 80)
            print("SUMMARY OF RESULTS:")
            print(f"  Final Accuracy:    {accuracy:.4f} {'✓' if accuracy >= 0.96 else '✗'}")
            print(f"  Final AUC:         {auc:.4f} {'✓' if auc >= 0.96 else '✗'}")
            print(f"  Sensitivity:       {sensitivity:.4f} {'✓' if sensitivity >= 0.95 else '✗'}")
            print(f"  Specificity:       {specificity:.4f} {'✓' if specificity >= 0.95 else '✗'}")
            
            if accuracy >= 0.96 and auc >= 0.96 and sensitivity >= 0.95 and specificity >= 0.95:
                print("\n🎉 CONGRATULATIONS! Model meets all clinical requirements!")
                print("   The system is ready for clinical validation studies.")
            else:
                print("\n⚠️  Model needs further improvement for clinical deployment.")
                print("   Check the clinical report for specific recommendations.")
            
            print(f"\n📁 All results saved to: {CONFIG['OUTPUT_DIR']}/")
            print("   • clinical_report.txt - Detailed clinical validation report")
            print("   • ensemble_config.json - Ensemble model configuration")
            print("   • *.h5 - Trained model files")
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nExiting pipeline. To run, execute: python clinical_pipeline.py")