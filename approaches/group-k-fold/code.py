# Requirements:
# pip install numpy pandas tensorflow scikit-learn xgboost kagglehub wfdb scipy pywt tqdm

import ast
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import wfdb
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers
from tqdm import tqdm  # Add this for progress bars
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 64,
    'EPOCHS': 40,
    'PATIENCE': 8,
    'LEARNING_RATE': 1e-3,
    'OUTPUT_DIR': './ecg_final_results',
    'INPUT_LENGTH': 1000,
    'NUM_LEADS': 12,
    'TEST_PATIENT_FRACTION': 0.15,
    'N_FOLDS': 5,
    'THRESHOLD_OBJECTIVE': 'accuracy',  # 'accuracy' or 'f1'
    'USE_SIMPLE_FEATURES': True,  # Use simpler features to speed up
    'MAX_WORKERS': 4,  # Reduce if memory issues
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ----------------------------
# DATA LOADING (patient-wise) - No changes needed
# ----------------------------
def load_data_simple():
    """Load PTB-XL, return X, y, patient_ids"""
    print("📥 Loading PTB-XL dataset...")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    csv_path = None
    data_root = None
    for root, _, files in os.walk(path):
        if 'ptbxl_database.csv' in files:
            csv_path = os.path.join(root, 'ptbxl_database.csv')
            data_root = root
            break
    if csv_path is None:
        raise FileNotFoundError("ptbxl_database.csv not found in dataset path")

    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def get_diag(codes):
        diag = []
        for c in codes:
            if c in scp_df.index:
                diag.append(scp_df.loc[c].diagnostic_class)
        return list(set(diag))

    df['diagnostic'] = df.scp_codes.apply(get_diag)
    df['label'] = df['diagnostic'].apply(lambda x: 1 if 'MI' in x else (0 if 'NORM' in x else -1))
    df = df[df['label'] != -1].reset_index()

    X, y, patient_ids = [], [], []

    def load_one(row):
        try:
            filepath = os.path.join(data_root, row['filename_lr'])
            rec = wfdb.rdrecord(filepath)
            sig = rec.p_signal  # (n_samples, 12)
            L = CONFIG['INPUT_LENGTH']
            if sig.shape[0] >= L:
                sig = sig[:L, :]
            else:
                pad = np.zeros((L - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
                sig = np.vstack([sig, pad])
            sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
            return sig.astype(np.float32)
        except Exception:
            return None

    rows = df.to_dict('records')
    batch = 500
    for i in range(0, len(rows), batch):
        sub = rows[i:i+batch]
        with ThreadPoolExecutor(max_workers=8) as ex:
            signals = list(ex.map(load_one, sub))
        for s, r in zip(signals, sub):
            if s is None: 
                continue
            if s.shape != (CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']):
                continue
            X.append(s)
            y.append(r['label'])
            patient_ids.append(r.get('patient_id', r.get('patientadx', None)))
        print(f"  Loaded {len(X)}/{len(rows)} signals...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    patient_ids = np.array(patient_ids)
    print(f"✅ Successfully loaded {len(X)} signals from {len(np.unique(patient_ids))} patients")
    return X, y, patient_ids

# ----------------------------
# PATIENT-WISE SPLIT - No changes
# ----------------------------
def patient_wise_split(X, y, patient_ids, test_fraction=0.15):
    unique_patients = np.unique(patient_ids)
    rng = np.random.default_rng(CONFIG['SEED'])
    rng.shuffle(unique_patients)
    n_test = max(1, int(len(unique_patients) * test_fraction))
    test_patients = set(unique_patients[:n_test])
    train_idx = [i for i, p in enumerate(patient_ids) if p not in test_patients]
    test_idx  = [i for i, p in enumerate(patient_ids) if p in test_patients]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_patient_ids = patient_ids[train_idx]
    test_patient_ids = patient_ids[test_idx]
    return X_train, X_test, y_train, y_test, train_patient_ids, test_patient_ids

# ----------------------------
# AUGMENT (training only) - No changes
# ----------------------------
def augment_data_simple(X, y):
    X_aug, y_aug = [], []
    for sig, label in zip(X, y):
        X_aug.append(sig); y_aug.append(label)
        noise = np.random.normal(0, 0.01, sig.shape).astype(np.float32)
        X_aug.append(sig + noise); y_aug.append(label)
        if label == 1:
            scale = np.random.uniform(0.95, 1.05)
            L = CONFIG['INPUT_LENGTH']
            idx = np.clip((np.arange(L) * scale).astype(int), 0, L-1)
            X_aug.append(sig[idx]); y_aug.append(label)
    idx = np.random.permutation(len(X_aug))
    return np.array(X_aug, dtype=np.float32)[idx], np.array(y_aug, dtype=np.int64)[idx]

# ----------------------------
# SIMPLIFIED FEATURE ENGINEERING (for XGBoost)
# ----------------------------
def extract_features_simple(X, fs=100):
    """Simplified feature extraction to speed up computation"""
    feats = []
    print("  Extracting features...")
    
    for sig in tqdm(X, desc="Feature extraction", leave=False):
        v = []
        # Basic statistical features for each lead
        for lead in range(min(6, sig.shape[1])):  # Use only first 6 leads to speed up
            s = sig[:, lead]
            v += [
                np.mean(s), np.std(s), np.max(s), np.min(s),
                np.median(s), stats.skew(s), stats.kurtosis(s),
                np.sum(np.abs(np.diff(s)))
            ]
        
        # Add some global features
        v += [
            np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
            np.percentile(sig, 25), np.percentile(sig, 75)
        ]
        
        feats.append(v)
    
    return np.array(feats, dtype=np.float32)

def extract_features_improved(X, fs=100):
    """Use simplified version if configured"""
    if CONFIG['USE_SIMPLE_FEATURES']:
        return extract_features_simple(X, fs)
    
    # Original implementation (slower)
    feats = []
    print("  Extracting features (full)...")
    
    for sig in tqdm(X, desc="Feature extraction", leave=False):
        v = []
        # per-lead basic stats
        for lead in range(sig.shape[1]):
            s = sig[:, lead]
            v += [
                np.mean(s), np.std(s), np.max(s), np.min(s),
                np.median(s), stats.skew(s), stats.kurtosis(s),
                np.sum(np.abs(np.diff(s)))
            ]
            # spectral for this lead
            try:
                f, Pxx = signal.welch(s, fs=fs, nperseg=min(256, len(s)))
                bands = [(0.5,4),(4,8),(8,13),(13,30)]
                for (lo, hi) in bands:
                    idx = np.logical_and(f >= lo, f <= hi)
                    v.append(np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0)
            except:
                v += [0.0] * 4
            # wavelet energies (per-lead)
            try:
                coeffs = pywt.wavedec(s, 'db4', level=4)
                energies = [np.sum(c**2) for c in coeffs[:4]]  # limit to first 4
                v += energies
            except:
                v += [0.0] * 4
        # inter-lead correlations summary
        try:
            corr = np.corrcoef(sig.T)
            iu = np.triu_indices(sig.shape[1], k=1)
            cvals = corr[iu]
            v += [np.mean(cvals), np.std(cvals), np.max(cvals)]
        except:
            v += [0.0, 0.0, 0.0]
        
        feats.append(v)
    
    return np.array(feats, dtype=np.float32)

# ----------------------------
# MODELS (builders) - No changes
# ----------------------------
def build_simple_cnn():
    inputs = layers.Input(shape=(CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']))
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, out, name='SimpleCNN')
    model.compile(optimizer=optimizers.Adam(CONFIG['LEARNING_RATE']), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def build_inception_simple():
    inputs = layers.Input(shape=(CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']))
    def im(x):
        b1 = layers.Conv1D(32,1,padding='same',activation='relu')(x)
        b2 = layers.Conv1D(32,3,padding='same',activation='relu')(x)
        b3 = layers.Conv1D(32,5,padding='same',activation='relu')(x)
        b4 = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        b4 = layers.Conv1D(32,1,padding='same',activation='relu')(b4)
        return layers.Concatenate()([b1,b2,b3,b4])
    x = layers.Conv1D(64,7,padding='same',activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = im(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = im(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, out, name='InceptionSimple')
    model.compile(optimizer=optimizers.Adam(CONFIG['LEARNING_RATE']), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def build_resnet_simple():
    def rb(x, filters):
        s = x
        x = layers.Conv1D(filters,3,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters,3,padding='same')(x)
        x = layers.BatchNormalization()(x)
        if s.shape[-1] != filters:
            s = layers.Conv1D(filters,1,padding='same')(s)
        x = layers.Add()([x,s])
        x = layers.Activation('relu')(x)
        return x
    inputs = layers.Input(shape=(CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']))
    x = layers.Conv1D(64,7,padding='same',activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = rb(x,64); x = layers.MaxPooling1D(2)(x)
    x = rb(x,128); x = layers.MaxPooling1D(2)(x)
    x = rb(x,256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, out, name='ResNetSimple')
    model.compile(optimizer=optimizers.Adam(CONFIG['LEARNING_RATE']), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# ----------------------------
# TRAIN SINGLE DL MODEL with progress
# ----------------------------
def train_dl_model(builder, X_tr, y_tr, X_val, y_val, name, epochs=CONFIG['EPOCHS']):
    print(f"    Training {name}...")
    model = builder()
    cb = [
        # callbacks.EarlyStopping(monitor='val_auc', patience=CONFIG['PATIENCE'], mode='max', restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=max(2, CONFIG['PATIENCE']//2), min_lr=1e-6, mode='max', verbose=0),
        callbacks.Callback()  # Custom callback for progress
    ]
    
    # Custom callback to show progress
    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                print(f"      {name}: Epoch {epoch+1}/{epochs}, AUC: {logs.get('auc', 0):.4f}, Val AUC: {logs.get('val_auc', 0):.4f}")
    
    cb.append(ProgressCallback())
    
    history = model.fit(
        X_tr, y_tr, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=CONFIG['BATCH_SIZE'], 
        callbacks=cb, 
        verbose=0
    )
    
    # Get best epoch
    best_epoch = np.argmax(history.history['val_auc'])
    print(f"    {name} finished: Best epoch {best_epoch+1}, Val AUC: {history.history['val_auc'][best_epoch]:.4f}")
    
    return model

# ----------------------------
# STACKING PIPELINE (GroupKFold) with better progress
# ----------------------------
def stacking_pipeline(X_train, y_train, train_patient_ids, X_test, y_test):
    print("\n🚀 Starting stacking pipeline...")
    print(f"   Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"   Using {CONFIG['N_FOLDS']} folds with patient-wise grouping")
    
    n_models = 4
    model_names = ['CNN', 'Inception', 'ResNet', 'XGBoost']
    oof_preds = {name: np.zeros(len(X_train), dtype=np.float32) for name in model_names}
    test_preds_folds = {name: [] for name in model_names}

    # features for XGBoost for full train/test
    print("\n📊 Extracting features for XGBoost...")
    start_time = time.time()
    X_train_feats_full = extract_features_improved(X_train)
    X_test_feats = extract_features_improved(X_test)
    print(f"   Feature extraction completed in {time.time()-start_time:.1f} seconds")

    gkf = GroupKFold(n_splits=CONFIG['N_FOLDS'])
    fold = 0
    
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_patient_ids):
        fold += 1
        print(f"\n{'='*50}")
        print(f"📂 Fold {fold}/{CONFIG['N_FOLDS']}")
        print(f"{'='*50}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Show fold statistics
        print(f"   Train: {len(X_tr)} samples ({np.sum(y_tr)} MI, {len(y_tr)-np.sum(y_tr)} NORM)")
        print(f"   Val: {len(X_val)} samples ({np.sum(y_val)} MI, {len(y_val)-np.sum(y_val)} NORM)")
        
        # augment training only
        print("   Augmenting training data...")
        start_aug = time.time()
        X_tr_aug, y_tr_aug = augment_data_simple(X_tr, y_tr)
        print(f"   Augmented to {len(X_tr_aug)} samples ({time.time()-start_aug:.1f}s)")

        # Train DL models on this fold
        print("\n   Training deep learning models...")
        
        # Train models sequentially with progress
        m_cnn = train_dl_model(build_simple_cnn, X_tr_aug, y_tr_aug, X_val, y_val, f'CNN_fold{fold}')
        m_inc = train_dl_model(build_inception_simple, X_tr_aug, y_tr_aug, X_val, y_val, f'Inception_fold{fold}')
        m_res = train_dl_model(build_resnet_simple, X_tr_aug, y_tr_aug, X_val, y_val, f'ResNet_fold{fold}')

        # XGBoost features for fold
        print("\n   Training XGBoost...")
        X_tr_feats = X_train_feats_full[train_idx]
        X_val_feats = X_train_feats_full[val_idx]
        scaler = StandardScaler().fit(X_tr_feats)
        X_tr_feats_s = scaler.transform(X_tr_feats)
        X_val_feats_s = scaler.transform(X_val_feats)
        
        xgb = XGBClassifier(
            n_estimators=200,  # Reduced for speed
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            n_jobs=-1, 
            random_state=CONFIG['SEED'], 
            use_label_encoder=False, 
            eval_metric='auc',
            verbosity=0
        )
        xgb.fit(X_tr_feats_s, y_tr, eval_set=[(X_val_feats_s, y_val)], verbose=0)
        xgb_auc = roc_auc_score(y_val, xgb.predict_proba(X_val_feats_s)[:, 1])
        print(f"    XGBoost: Val AUC: {xgb_auc:.4f}")

        # OOF predictions for this fold
        print("   Making predictions...")
        oof_preds['CNN'][val_idx] = m_cnn.predict(X_val, verbose=0).flatten()
        oof_preds['Inception'][val_idx] = m_inc.predict(X_val, verbose=0).flatten()
        oof_preds['ResNet'][val_idx] = m_res.predict(X_val, verbose=0).flatten()
        oof_preds['XGBoost'][val_idx] = xgb.predict_proba(X_val_feats_s)[:, 1]

        # Test predictions from this fold (append)
        test_feats_s = scaler.transform(X_test_feats)
        test_preds_folds['CNN'].append(m_cnn.predict(X_test, verbose=0).flatten())
        test_preds_folds['Inception'].append(m_inc.predict(X_test, verbose=0).flatten())
        test_preds_folds['ResNet'].append(m_res.predict(X_test, verbose=0).flatten())
        test_preds_folds['XGBoost'].append(xgb.predict_proba(test_feats_s)[:, 1])

        # quick fold metrics
        print(f"\n   Fold {fold} validation metrics:")
        for name in model_names:
            preds = oof_preds[name][val_idx]
            auc_val = roc_auc_score(y_val, preds)
            print(f"     {name}: AUC = {auc_val:.4f}")

    # After folds: train meta-learner on OOF preds
    print("\n" + "="*50)
    print("🧠 Training meta-learner (stacking)...")
    stacked_oof = np.vstack([oof_preds[n] for n in model_names]).T
    meta = LogisticRegression(max_iter=2000, solver='lbfgs')
    meta.fit(stacked_oof, y_train)
    print("   Meta-learner trained.")

    # Average test preds across folds per model
    test_preds_avg = {}
    for name in model_names:
        test_preds_avg[name] = np.mean(np.vstack(test_preds_folds[name]), axis=0)

    # stacked test input
    stacked_test = np.vstack([test_preds_avg[n] for n in model_names]).T
    stacked_test_probs = meta.predict_proba(stacked_test)[:, 1]

    # threshold tuning on OOF stacked probs
    print("\n📈 Tuning threshold...")
    stacked_oof_probs = meta.predict_proba(stacked_oof)[:, 1]
    best_thr = 0.5
    best_score = -1
    thresholds = np.linspace(0.3, 0.7, 41)  # Reduced from 81 to 41 for speed
    
    for thr in thresholds:
        if CONFIG['THRESHOLD_OBJECTIVE'] == 'accuracy':
            s = accuracy_score(y_train, (stacked_oof_probs >= thr).astype(int))
        else:
            s = f1_score(y_train, (stacked_oof_probs >= thr).astype(int))
        if s > best_score:
            best_score = s
            best_thr = thr
    
    print(f"   Chosen threshold ({CONFIG['THRESHOLD_OBJECTIVE']}): {best_thr:.3f} (score={best_score:.4f})")

    # evaluate on test
    y_pred_test = (stacked_test_probs >= best_thr).astype(int)
    ensemble_auc = roc_auc_score(y_test, stacked_test_probs)
    ensemble_acc = accuracy_score(y_test, y_pred_test)
    ensemble_f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"\n{'='*50}")
    print("🎯 Final ensemble test set performance:")
    print(f"   AUC: {ensemble_auc:.4f}")
    print(f"   Accuracy: {ensemble_acc:.4f}")
    print(f"   F1 Score: {ensemble_f1:.4f}")
    print(f"   Confusion matrix:\n{cm}")
    print(f"{'='*50}")
    
    return {
        'meta': meta,
        'oof_preds': oof_preds,
        'test_preds_folds': test_preds_folds,
        'test_preds_avg': test_preds_avg,
        'stacked_test_probs': stacked_test_probs,
        'threshold': best_thr,
        'metrics': {'auc': ensemble_auc, 'acc': ensemble_acc, 'f1': ensemble_f1, 'cm': cm}
    }

# ----------------------------
# SIMPLIFIED FINAL RETRAIN (optional - can skip if taking too long)
# ----------------------------
def retrain_full_and_predict(X_train, y_train, train_patient_ids, X_test):
    print("\n🔄 Retraining on full training set...")
    
    # Train only the best performing model or a subset to save time
    print("   Training final CNN model...")
    X_train_aug, y_train_aug = augment_data_simple(X_train, y_train)
    m_cnn = train_dl_model(build_simple_cnn, X_train_aug, y_train_aug, X_train, y_train, 'CNN_final')
    
    # Train XGBoost as well
    print("   Training final XGBoost model...")
    X_train_feats = extract_features_simple(X_train)
    X_test_feats = extract_features_simple(X_test)
    scaler = StandardScaler().fit(X_train_feats)
    X_train_feats_s = scaler.transform(X_train_feats)
    X_test_feats_s = scaler.transform(X_test_feats)
    
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=CONFIG['SEED'],
        use_label_encoder=False,
        eval_metric='auc',
        verbosity=0
    )
    xgb.fit(X_train_feats_s, y_train, verbose=0)
    
    # collect predictions from both models
    preds = {
        'CNN': m_cnn.predict(X_test, verbose=0).flatten(),
        'XGBoost': xgb.predict_proba(X_test_feats_s)[:, 1]
    }
    
    return preds

# ----------------------------
# MAIN with error handling
# ----------------------------
def main():
    try:
        print("🏁 Starting ECG classification pipeline...")
        
        # Load data
        X, y, patient_ids = load_data_simple()
        
        # Split data
        X_train, X_test, y_train, y_test, train_patient_ids, test_patient_ids = patient_wise_split(
            X, y, patient_ids, CONFIG['TEST_PATIENT_FRACTION']
        )
        print(f"\n📊 Dataset split:")
        print(f"   Train: {len(X_train)} samples from {len(np.unique(train_patient_ids))} patients")
        print(f"   Test: {len(X_test)} samples from {len(np.unique(test_patient_ids))} patients")
        print(f"   MI/NORM ratio - Train: {np.sum(y_train)/len(y_train):.3f}, Test: {np.sum(y_test)/len(y_test):.3f}")

        # Run stacking pipeline
        results = stacking_pipeline(X_train, y_train, train_patient_ids, X_test, y_test)

        # Optional: Retrain on full training set
        print("\n" + "="*50)
        print("🚀 Final evaluation...")
        
        final_base_preds = retrain_full_and_predict(X_train, y_train, train_patient_ids, X_test)
        
        # Use simple stacking with available models
        available_models = ['CNN', 'XGBoost']
        stacked_final_test = np.vstack([final_base_preds[n] for n in available_models]).T
        
        # Create a simple meta-learner
        from sklearn.ensemble import VotingClassifier
        meta_simple = LogisticRegression(max_iter=1000, solver='lbfgs')
        
        # Create simple OOF-like predictions for meta training
        # We'll use the OOF predictions from the stacking pipeline for CNN and XGBoost
        simple_oof = np.vstack([results['oof_preds']['CNN'], results['oof_preds']['XGBoost']]).T
        meta_simple.fit(simple_oof, y_train)
        
        final_probs = meta_simple.predict_proba(stacked_final_test)[:,1]
        final_preds = (final_probs >= results['threshold']).astype(int)

        # Final metrics
        auc_final = roc_auc_score(y_test, final_probs)
        acc_final = accuracy_score(y_test, final_preds)
        f1_final = f1_score(y_test, final_preds)
        cm = confusion_matrix(y_test, final_preds)

        print("\n" + "="*50)
        print("🏆 FINAL RESULTS")
        print("="*50)
        print(f"AUC:  {auc_final:.4f}")
        print(f"Accuracy: {acc_final:.4f}")
        print(f"F1 Score: {f1_final:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Threshold used: {results['threshold']:.3f}")
        print("="*50)

        # Save summary
        out = os.path.join(CONFIG['OUTPUT_DIR'], 'stacking_results.txt')
        with open(out, 'w') as f:
            f.write("FINAL RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"AUC: {auc_final:.4f}\n")
            f.write(f"Accuracy: {acc_final:.4f}\n")
            f.write(f"F1 Score: {f1_final:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Threshold: {results['threshold']:.3f}\n")
            f.write("="*50 + "\n")
            f.write(f"\nTraining Configuration:\n")
            for key, value in CONFIG.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\n💾 Results saved to {out}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()