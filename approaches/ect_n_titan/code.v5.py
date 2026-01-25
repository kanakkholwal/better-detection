# TITAN + Simple pipeline — Patient-wise splits (both pipelines preserved)
# Requirements:
# pip install numpy pandas tensorflow scikit-learn xgboost kagglehub wfdb

import ast
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import numpy as np

# CRITICAL FIX: Set pandas options BEFORE importing wfdb
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers
from xgboost import XGBClassifier

pd.options.future.infer_string = False
# pd.options.mode.dtype_backend = "numpy"
import wfdb

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 64,
    'EPOCHS': 40,
    'PATIENCE': 12,
    'LEARNING_RATE': 1e-3,
    'OUTPUT_DIR': './ecg_final_results',
    'INPUT_LENGTH': 1000,
    'NUM_LEADS': 12,
    # Fractions are of the whole dataset (patient-level)
    'TEST_PATIENT_FRACTION': 0.15,
    'VAL_PATIENT_FRACTION': 0.15
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# ----------------------------
# DATA LOADING (patient-wise)
# ----------------------------
def load_data_simple():
    """Load PTB-XL, return X, y, patient_ids"""
    print("Loading PTB-XL dataset...")
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

    # Read CSV - dtype_backend already set globally
    df = pd.read_csv(csv_path)

    # Convert ecg_id to string and set as index
    if 'ecg_id' in df.columns:
        df['ecg_id'] = df['ecg_id'].astype(str)
    df = df.set_index('ecg_id')

    # Parse scp_codes safely
    if 'scp_codes' in df.columns:
        df['scp_codes'] = df['scp_codes'].astype(str).apply(ast.literal_eval)

    # Read scp_statements
    scp_df = pd.read_csv(os.path.join(data_root, 'scp_statements.csv'))
    
    # Set index for scp_df
    first_col_name = scp_df.columns[0]
    scp_df[first_col_name] = scp_df[first_col_name].astype(str)
    scp_df = scp_df.set_index(first_col_name)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def get_diag(codes):
        diag = []
        for c in codes:
            c = str(c)
            if c in scp_df.index:
                diag.append(scp_df.loc[c].diagnostic_class)
        return list(set(diag))

    df['diagnostic'] = df.scp_codes.apply(get_diag)
    df['label'] = df['diagnostic'].apply(lambda x: 1 if 'MI' in x else (0 if 'NORM' in x else -1))
    df = df[df['label'] != -1].reset_index()

    # Load ECG signals
    def load_ecg(row):
        filename = os.path.join(data_root, row['filename_hr'])
        try:
            record = wfdb.rdsamp(filename.replace('.dat', ''))
            sig = record[0]
            if sig.shape[0] >= CONFIG['INPUT_LENGTH']:
                sig = sig[:CONFIG['INPUT_LENGTH'], :]
            else:
                pad = CONFIG['INPUT_LENGTH'] - sig.shape[0]
                sig = np.vstack([sig, np.zeros((pad, sig.shape[1]))])
            return sig
        except Exception:
            return None

    print("Loading ECG signals (parallel)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        signals = list(executor.map(load_ecg, [row for _, row in df.iterrows()]))

    valid_idx = [i for i, s in enumerate(signals) if s is not None]
    X = np.array([signals[i] for i in valid_idx], dtype=np.float32)
    y = df.iloc[valid_idx]['label'].values.astype(np.int64)
    patient_ids = df.iloc[valid_idx]['patient_id'].values.astype(str)

    print(f"Loaded {len(X)} samples from {len(np.unique(patient_ids))} patients")
    return X, y, patient_ids

# ----------------------------
# STRICT PATIENT-WISE SPLIT (no leakage)
# ----------------------------
def patient_wise_split_strict(X, y, patient_ids, test_fraction=0.15, val_fraction=0.15, seed=None):
    if seed is None:
        seed = CONFIG['SEED']
    rng = np.random.default_rng(seed)
    unique_patients = np.array(sorted(np.unique(patient_ids)))
    rng.shuffle(unique_patients)

    n_pat = len(unique_patients)
    n_test = max(1, int(np.round(n_pat * test_fraction)))
    n_val = max(1, int(np.round(n_pat * val_fraction)))

    test_patients = set(unique_patients[:n_test])
    val_patients = set(unique_patients[n_test:n_test + n_val])
    train_patients = set(unique_patients[n_test + n_val:])

    train_idx = [i for i, p in enumerate(patient_ids) if p in train_patients]
    val_idx = [i for i, p in enumerate(patient_ids) if p in val_patients]
    test_idx = [i for i, p in enumerate(patient_ids) if p in test_patients]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test, train_patients, val_patients, test_patients

# ----------------------------
# DATA AUGMENTATION (training-only)
# ----------------------------
def augment_data_simple(X, y):
    X_aug, y_aug = [], []
    for sig, label in zip(X, y):
        X_aug.append(sig)
        y_aug.append(label)
        noise = np.random.normal(0, 0.01, sig.shape).astype(np.float32)
        X_aug.append(sig + noise)
        y_aug.append(label)
        if label == 1:
            scale = np.random.uniform(0.95, 1.05)
            L = CONFIG['INPUT_LENGTH']
            idx = np.clip((np.arange(L) * scale).astype(int), 0, L - 1)
            X_aug.append(sig[idx])
            y_aug.append(label)
    idx = np.random.permutation(len(X_aug))
    return np.array(X_aug, dtype=np.float32)[idx], np.array(y_aug, dtype=np.int64)[idx]

# ----------------------------
# MODELS (kept from your original)
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
    x = layers.Dense(128, activation='relu', name='penultimate')(x)
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
    print(f"\nTraining {model_name}...")
    model = model_builder()
    cb = [
        # callbacks.EarlyStopping(monitor='val_auc', patience=CONFIG['PATIENCE'], mode='max', restore_best_weights=True, verbose=1),
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
    print(f"{model_name} - Best epoch: {best_epoch}, Val AUC: {best_auc:.4f}, Val Acc: {best_acc:.4f}")
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
# PIPELINE - SIMPLE (patient-wise correct)
# ----------------------------
def main_simple():
    print("="*60)
    print("ECG SIMPLE PIPELINE - patient-wise (strict)")
    print("="*60)

    X, y, patient_ids = load_data_simple()
    X_train, X_val, X_test, y_train, y_val, y_test, train_p, val_p, test_p = patient_wise_split_strict(
        X, y, patient_ids,
        test_fraction=CONFIG['TEST_PATIENT_FRACTION'],
        val_fraction=CONFIG['VAL_PATIENT_FRACTION'],
        seed=CONFIG['SEED']
    )
    print(f"  Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Unique patients - train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}")

    # AUGMENT only training set
    X_train_aug, y_train_aug = augment_data_simple(X_train, y_train)
    print(f"  After augment: train {len(X_train_aug)}")

    models_dict = {}
    histories = {}

    m1, h1 = train_model_simple(build_simple_cnn, X_train_aug, y_train_aug, X_val, y_val, "SimpleCNN")
    models_dict['CNN'] = m1
    histories['CNN'] = h1

    m2, h2 = train_model_simple(build_inception_simple, X_train_aug, y_train_aug, X_val, y_val, "Inception")
    models_dict['Inception'] = m2 
    histories['Inception'] = h2

    m3, h3 = train_model_simple(build_resnet_simple, X_train_aug, y_train_aug, X_val, y_val, "ResNet")
    models_dict['ResNet'] = m3
    histories['ResNet'] = h3

    # XGBoost on extracted features (use training augmented set for training XGB)
    X_train_feats = extract_features(X_train_aug)
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
    xgb.fit(X_train_feats_s, y_train_aug, eval_set=[(X_val_feats_s, y_val)], verbose=0)
    models_dict['XGBoost'] = xgb

    # predictions
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

    perf = {n: roc_auc_score(y_val, p) for n, p in preds_val.items()}
    total = sum(perf.values())
    weights = {n: (perf[n] / total) for n in perf} if total > 0 else {n: 1.0/len(perf) for n in perf}
    print("Ensemble Weights:")
    for n, w in weights.items():
        print(f"  {n}: {w:.3f}")

    ensemble_test = sum(preds_test[n] * weights[n] for n in preds_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_test)
    ensemble_acc = accuracy_score(y_test, (ensemble_test > 0.5).astype(int))
    cm = confusion_matrix(y_test, (ensemble_test > 0.5).astype(int))

    # save minimal summary
    out_path = os.path.join(CONFIG['OUTPUT_DIR'], "simple_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Ensemble AUC: {ensemble_auc:.4f}\nAccuracy: {ensemble_acc:.4f}\nConfusion: {cm.tolist()}\n")
        f.write("Weights:\n")
        for n, w in weights.items():
            f.write(f"{n}: {w:.3f}\n")
    print(f"Results saved to {out_path}")
    return ensemble_acc, ensemble_auc

# ----------------------------
# TITAN pipeline implemented using patient-wise split
# ----------------------------
def run_titan_pipeline():
    """
    Runs a Titan-like ensemble using strict patient-wise test/val splits.
    Re-uses builders: Inception, ResNet, SimpleCNN and XGBoost on Inception embeddings.
    """
    print("="*60)
    print("RUNNING TITAN PIPELINE (patient-wise split)")
    print("="*60)

    X, y, patient_ids = load_data_simple()
    X_train, X_val, X_test, y_train, y_val, y_test, train_p, val_p, test_p = patient_wise_split_strict(
        X, y, patient_ids,
        test_fraction=CONFIG['TEST_PATIENT_FRACTION'],
        val_fraction=CONFIG['VAL_PATIENT_FRACTION'],
        seed=CONFIG['SEED']
    )
    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Patients - train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}")

    # augmentation on train only
    X_train_aug, y_train_aug = augment_data_simple(X_train, y_train)
    print(f"After augmentation: train {len(X_train_aug)}")

    # Train heavy models (Inception anchor + ResNet + CNN)
    models_dict = {}
    print("\nTraining Inception (anchor)...")
    m_incep, _ = train_model_simple(build_inception_simple, X_train_aug, y_train_aug, X_val, y_val, "Titan_Inception")
    models_dict['Inception'] = m_incep

    print("\nTraining ResNet (support)...")
    m_res, _ = train_model_simple(build_resnet_simple, X_train_aug, y_train_aug, X_val, y_val, "Titan_ResNet")
    models_dict['ResNet'] = m_res

    print("\nTraining Simple CNN (support)...")
    m_cnn, _ = train_model_simple(build_simple_cnn, X_train_aug, y_train_aug, X_val, y_val, "Titan_CNN")
    models_dict['CNN'] = m_cnn

    # Extract embeddings from Inception (penultimate named 'penultimate')
    try:
        embed_layer = None
        for layer in m_incep.layers[::-1]:
            if layer.name == 'penultimate':
                embed_layer = layer
                break
        if embed_layer is None:
            embed_layer = m_incep.layers[-2]
        embedding_model = models.Model(m_incep.input, embed_layer.output)
    except Exception:
        embedding_model = None

    # Prepare embeddings for XGBoost
    if embedding_model is not None:
        train_embeds = embedding_model.predict(X_train_aug, verbose=0)
        val_embeds = embedding_model.predict(X_val, verbose=0)
        test_embeds = embedding_model.predict(X_test, verbose=0)
    else:
        train_embeds = extract_features(X_train_aug)
        val_embeds = extract_features(X_val)
        test_embeds = extract_features(X_test)

    scaler = StandardScaler().fit(np.vstack([train_embeds, val_embeds]))
    train_embeds_s = scaler.transform(train_embeds)
    val_embeds_s = scaler.transform(val_embeds)
    test_embeds_s = scaler.transform(test_embeds)

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=CONFIG['SEED'],
        use_label_encoder=False,
        eval_metric='auc'
    )
    print("\nTraining XGBoost on embeddings...")
    xgb.fit(train_embeds_s, y_train_aug, eval_set=[(val_embeds_s, y_val)], verbose=0)
    models_dict['XGB_Embed'] = xgb

    # Predictions on validation and test
    preds_val = {}
    preds_test = {}
    for name, mdl in models_dict.items():
        if name == 'XGB_Embed':
            pv = mdl.predict_proba(val_embeds_s)[:, 1]
            pt = mdl.predict_proba(test_embeds_s)[:, 1]
        else:
            pv = mdl.predict(X_val, verbose=0).flatten()
            pt = mdl.predict(X_test, verbose=0).flatten()
        preds_val[name] = pv
        preds_test[name] = pt
        print(f"  {name}: Val AUC = {roc_auc_score(y_val, pv):.4f}, Test AUC = {roc_auc_score(y_test, pt):.4f}")

    # Weighted ensemble (favor higher val AUC)
    perf = {n: roc_auc_score(y_val, p) for n, p in preds_val.items()}
    total = sum(perf.values())
    weights = {n: (perf[n] / total) for n in perf} if total > 0 else {n: 1.0 / len(perf) for n in perf}

    print("\nEnsemble Weights (from val AUC):")
    for n, w in weights.items():
        print(f"  {n}: {w:.3f}")

    ensemble_test = sum(preds_test[n] * weights[n] for n in preds_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_test)
    ensemble_acc = accuracy_score(y_test, (ensemble_test > 0.5).astype(int))
    cm = confusion_matrix(y_test, (ensemble_test > 0.5).astype(int))

    # metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0
    print("\nCONFUSION MATRIX (TITAN)")
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")
    # save report
    out_path = os.path.join(CONFIG['OUTPUT_DIR'], "titan_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Ensemble AUC: {ensemble_auc:.4f}\nAccuracy: {ensemble_acc:.4f}\nConfusion: {cm.tolist()}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}\n")
        f.write("Weights:\n")
        for n, w in weights.items():
            f.write(f"{n}: {w:.3f}\n")
    print(f"\nTITAN results saved to {out_path}")

    print("\nFINAL TITAN METRICS")
    print(f"AUC:      {ensemble_auc:.4f}")
    print(f"Accuracy: {ensemble_acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}")
    return ensemble_acc, ensemble_auc

# ----------------------------
# ENTRY
# ----------------------------
if __name__ == "__main__":
    print("1: Run Simple Pipeline (patient-wise)\n2: Run Titan (patient-wise)\n3: Both")
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