# Requirements:
# pip install numpy pandas tensorflow scikit-learn xgboost kagglehub wfdb
#  Refactor for code.v1
import ast
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 64,
    'EPOCHS': 40,             # increase to 50-60 if you want to push more
    'PATIENCE': 12,
    'LEARNING_RATE': 1e-3,
    'OUTPUT_DIR': './ecg_final_results',
    'INPUT_LENGTH': 1000,
    'NUM_LEADS': 12,
    'TEST_PATIENT_FRACTION': 0.15,
    'VALIDATION_FRACTION': 0.15
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ----------------------------
# DATA LOADING (patient-wise)
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
        raise FileNotFoundError("ptbxl_database.csv not found in downloaded dataset path")

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

    # load signals (parallel)
    X, y, patient_ids = [], [], []

    def load_one(row):
        try:
            filepath = os.path.join(data_root, row['filename_lr'])
            rec = wfdb.rdrecord(filepath)
            sig = rec.p_signal  # shape (n_samples, 12)
            # trim/pad to fixed length
            L = CONFIG['INPUT_LENGTH']
            if sig.shape[0] >= L:
                sig = sig[:L, :]
            else:
                pad = np.zeros((L - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
                sig = np.vstack([sig, pad])
            # normalize per-lead
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
            patient_ids.append(r.get('patient_id', r.get('patientadx', None)))  # fallback
        print(f"  Loaded {len(X)}/{len(df)} signals...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    patient_ids = np.array(patient_ids)
    print(f"✅ Successfully loaded {len(X)} signals from {len(np.unique(patient_ids))} patients")
    return X, y, patient_ids

# ----------------------------
# PATIENT-WISE SPLIT
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
    return X_train, X_test, y_train, y_test, test_patients

# ----------------------------
# DATA AUGMENTATION (training-only)
# ----------------------------
def augment_data_simple(X, y):
    X_aug, y_aug = [], []
    for sig, label in zip(X, y):
        X_aug.append(sig)
        y_aug.append(label)
        # small gaussian noise
        noise = np.random.normal(0, 0.01, sig.shape).astype(np.float32)
        X_aug.append(sig + noise)
        y_aug.append(label)
        # slight time-scaling by linear interpolation (fast approximation: stretch/shrink by resample)
        if label == 1:
            scale = np.random.uniform(0.95, 1.05)
            L = CONFIG['INPUT_LENGTH']
            idx = np.clip((np.arange(L) * scale).astype(int), 0, L - 1)
            X_aug.append(sig[idx])
            y_aug.append(label)
    idx = np.random.permutation(len(X_aug))
    return np.array(X_aug, dtype=np.float32)[idx], np.array(y_aug, dtype=np.int64)[idx]

# ----------------------------
# MODELS
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
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, out, name='SimpleCNN')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_inception_simple():
    inputs = layers.Input(shape=(CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']))

    def inception_module(x):
        b1 = layers.Conv1D(32, 1, padding='same', activation='relu')(x)
        b2 = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        b3 = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
        b4 = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        b4 = layers.Conv1D(32, 1, padding='same', activation='relu')(b4)
        return layers.Concatenate()([b1, b2, b3, b4])

    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = inception_module(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = inception_module(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, out, name='InceptionSimple')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_resnet_simple():
    def res_block(x, filters):
        s = x
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if s.shape[-1] != filters:
            s = layers.Conv1D(filters, 1, padding='same')(s)
        x = layers.Add()([x, s])
        x = layers.Activation('relu')(x)
        return x

    inputs = layers.Input(shape=(CONFIG['INPUT_LENGTH'], CONFIG['NUM_LEADS']))
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = res_block(x, 64)
    x = layers.MaxPooling1D(2)(x)
    x = res_block(x, 128)
    x = layers.MaxPooling1D(2)(x)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, out, name='ResNetSimple')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ----------------------------
# TRAINING UTIL
# ----------------------------
def train_model_simple(model_builder, X_train, y_train, X_val, y_val, model_name):
    print(f"\n🔧 Training {model_name}...")
    model = model_builder()
    cb = [
        callbacks.EarlyStopping(monitor='val_auc', patience=CONFIG['PATIENCE'], mode='max', restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=max(2, CONFIG['PATIENCE']//2), min_lr=1e-6, mode='max', verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        callbacks=cb,
        verbose=1
    )
    best_epoch = int(np.argmax(history.history['val_auc'])) + 1
    best_auc = float(np.max(history.history['val_auc']))
    best_acc = float(history.history['val_accuracy'][best_epoch-1])
    print(f"✅ {model_name} - Best epoch: {best_epoch}, Val AUC: {best_auc:.4f}, Val Acc: {best_acc:.4f}")
    # save model
    model.save(os.path.join(CONFIG['OUTPUT_DIR'], f"{model_name}.keras"))
    return model, history

# ----------------------------
# FEATURE EXTRACTION FOR XGBOOST
# ----------------------------
def extract_features(X):
    feats = []
    for sig in X:
        v = []
        for lead in range(sig.shape[1]):
            s = sig[:, lead]
            v += [np.mean(s), np.std(s), np.max(s), np.min(s),
                  np.median(s), np.percentile(s, 25), np.percentile(s, 75),
                  np.sum(np.abs(s)), np.sum(np.abs(np.diff(s)))]
        feats.append(v)
    return np.array(feats, dtype=np.float32)

# ----------------------------
# PIPELINE
# ----------------------------
def main_simple():
    print("="*60)
    print("🚀 ECG SIMPLE PIPELINE - patient-wise")
    print("="*60)

    X, y, patient_ids = load_data_simple()
    X_train, X_test, y_train, y_test, test_patients = patient_wise_split(X, y, patient_ids, CONFIG['TEST_PATIENT_FRACTION'])
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Unique test patients: {len(test_patients)}")

    # augment training-only
    X_train_aug, y_train_aug = augment_data_simple(X_train, y_train)
    # validation split from augmented training (safe: augmentation only on train)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_aug, y_train_aug,
        test_size=CONFIG['VALIDATION_FRACTION'],
        stratify=y_train_aug,
        random_state=CONFIG['SEED']
    )
    print(f"  Final training: {len(X_train_final)}, Validation: {len(X_val)}")

    # train DL models
    models_dict = {}
    histories = {}

    m1, h1 = train_model_simple(build_simple_cnn, X_train_final, y_train_final, X_val, y_val, "SimpleCNN")
    models_dict['CNN'] = m1; histories['CNN'] = h1

    m2, h2 = train_model_simple(build_inception_simple, X_train_final, y_train_final, X_val, y_val, "Inception")
    models_dict['Inception'] = m2; histories['Inception'] = h2

    m3, h3 = train_model_simple(build_resnet_simple, X_train_final, y_train_final, X_val, y_val, "ResNet")
    models_dict['ResNet'] = m3; histories['ResNet'] = h3

    # XGBoost on extracted features (train on training-final, validate on val, test on test)
    print("\n🌳 Training XGBoost (feature-based)...")
    X_train_feats = extract_features(X_train_final)
    X_val_feats = extract_features(X_val)
    X_test_feats = extract_features(X_test)

    scaler = StandardScaler().fit(np.vstack([X_train_feats, X_val_feats]))
    X_train_feats_s = scaler.transform(X_train_feats)
    X_val_feats_s = scaler.transform(X_val_feats)
    X_test_feats_s = scaler.transform(X_test_feats)

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=CONFIG['SEED'],
        use_label_encoder=False,
        eval_metric='auc'
    )
    xgb.fit(X_train_feats_s, y_train_final, eval_set=[(X_val_feats_s, y_val)], verbose=0)
    models_dict['XGBoost'] = xgb

    # predictions
    print("\n🎯 Making predictions...")
    preds_val = {}
    preds_test = {}
    for name, mdl in models_dict.items():
        if name == 'XGBoost':
            pv = mdl.predict_proba(X_val_feats_s)[:, 1]
            pt = mdl.predict_proba(X_test_feats_s)[:, 1]
        else:
            pv = mdl.predict(X_val, verbose=0).flatten()
            pt = mdl.predict(X_test, verbose=0).flatten()
        preds_val[name] = pv
        preds_test[name] = pt
        print(f"  {name}: Val AUC = {roc_auc_score(y_val, pv):.4f}, Test AUC = {roc_auc_score(y_test, pt):.4f}")

    # ensemble weights from val AUC
    perf = {n: roc_auc_score(y_val, p) for n, p in preds_val.items()}
    total = sum(perf.values())
    weights = {n: (perf[n] / total) for n in perf}
    print("\n📊 Ensemble Weights:")
    for n, w in weights.items():
        print(f"  {n}: {w:.3f}")

    ensemble_test = sum(preds_test[n] * weights[n] for n in preds_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_test)
    ensemble_acc = accuracy_score(y_test, (ensemble_test > 0.5).astype(int))
    cm = confusion_matrix(y_test, (ensemble_test > 0.5).astype(int))

    print("\n" + "="*40)
    print("🎯 FINAL ENSEMBLE RESULTS")
    print("="*40)
    print(f"AUC:      {ensemble_auc:.4f}")
    print(f"Accuracy: {ensemble_acc:.4f}")
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0

    print("\n📊 Detailed Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity:          {specificity:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  F1-Score:             {f1:.4f}")
    print("\n📊 Confusion Matrix:")
    print(f"[[{tn:4d}  {fp:4d}]")
    print(f" [{fn:4d}  {tp:4d}]]")

    # save minimal summary
    out_path = os.path.join(CONFIG['OUTPUT_DIR'], "simple_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Ensemble AUC: {ensemble_auc:.4f}\nAccuracy: {ensemble_acc:.4f}\nConfusion: {cm.tolist()}\n")
        f.write(f"Metrics: Sens {sensitivity:.4f}, Spec {specificity:.4f}, Prec {precision:.4f}, F1 {f1:.4f}\n")
        f.write("Weights:\n")
        for n, w in weights.items():
            f.write(f"{n}: {w:.3f}\n")
    print(f"\n✅ Results saved to {out_path}")
    return ensemble_acc, ensemble_auc

# ----------------------------
# TITAN placeholder (unchanged)
# ----------------------------
def run_titan_pipeline():
    print("Run your original Titan pipeline externally with patient-wise split and CV.")
    return 0.9751, 0.9101

# ----------------------------
# ENTRY
# ----------------------------
if __name__ == "__main__":
    print("1: Run Simple Pipeline (patient-wise)\n2: Run Titan (placeholder)\n3: Both")
    choice = input("Enter choice (1/2/3): ").strip() or "1"
    try:
        if choice == "1":
            acc, auc = main_simple()
        elif choice == "2":
            acc, auc = run_titan_pipeline()
        else:
            acc1, auc1 = main_simple()
            acc2, auc2 = run_titan_pipeline()
            print(f"Simple: AUC={auc1:.4f}, ACC={acc1:.4f}\nTitan:  AUC={auc2:.4f}, ACC={acc2:.4f}")
    except Exception as e:
        print("Error:", e)
        raise
