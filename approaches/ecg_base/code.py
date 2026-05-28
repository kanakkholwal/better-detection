# ==============================================================================
# PTB-XL MI Detection — HIGH ACCURACY / NO LEAKAGE / ECG-CORRECT
# ==============================================================================

import os, ast, numpy as np, pandas as pd, wfdb, tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import kagglehub

# ---------------- SYSTEM ----------------
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()

# ---------------- CONFIG ----------------
FS = 250
SECONDS = 10
MAX_LEN = FS * SECONDS
BATCH = 8
EPOCHS = 50
LR = 3e-4

# ---------------- DATA ----------------
base = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")

def find(root, name):
    for r, _, f in os.walk(root):
        if name in f:
            return os.path.join(r, name)
    raise FileNotFoundError(name)

db = pd.read_csv(find(base, "ptbxl_database.csv"), index_col="ecg_id")
scp = pd.read_csv(find(base, "scp_statements.csv"), index_col=0)
scp = scp[scp.diagnostic == 1]

db.scp_codes = db.scp_codes.apply(ast.literal_eval)

def map_label(codes):
    for k in codes:
        if k in scp.index and scp.loc[k].diagnostic_class == "MI":
            return 1
    return 0

db["target"] = db.scp_codes.apply(map_label)

DATA_ROOT = os.path.dirname(find(base, "ptbxl_database.csv"))

# OFFICIAL SPLIT (MANDATORY)
train_df = db[db.strat_fold <= 8]
val_df   = db[db.strat_fold == 9]
test_df  = db[db.strat_fold == 10]

# ---------------- SIGNAL LOADER ----------------
def load_ecg(path):
    sig = wfdb.rdrecord(os.path.join(DATA_ROOT, path)).p_signal
    sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
    sig = sig[:MAX_LEN] if len(sig) >= MAX_LEN else np.pad(sig, ((0, MAX_LEN-len(sig)), (0,0)))
    return sig.astype(np.float32)

def gen(df):
    for _, r in df.iterrows():
        yield load_ecg(r.filename_lr), r.target

def make_ds(df, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: gen(df),
        output_signature=(
            tf.TensorSpec((MAX_LEN,12), tf.float32),
            tf.TensorSpec((), tf.int32)
        )
    )
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df, True)
val_ds   = make_ds(val_df)
test_ds  = make_ds(test_df)

# ---------------- MODEL ----------------
def residual(x, f, s=1):
    sc = x
    x = layers.Conv1D(f, 7, s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(f, 7, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if sc.shape[-1] != f or s != 1:
        sc = layers.Conv1D(f, 1, s, padding="same", use_bias=False)(sc)
    return layers.ReLU()(layers.Add()([x, sc]))
class EpochIndicator(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        auc = logs.get("val_auc", 0.0)
        self.best = max(self.best, auc)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"loss={logs['loss']:.4f} | "
            f"auc={logs['auc']:.4f} | "
            f"val_auc={auc:.4f} | "
            f"best_val_auc={self.best:.4f} | "
            f"lr={lr:.2e}"
        )

class LeadAttention(layers.Layer):
    def build(self, s):
        self.fc1 = layers.Dense(32, activation="relu")
        self.fc2 = layers.Dense(s[-1], activation="sigmoid")
    def call(self, x):
        w = tf.reduce_mean(x, axis=1)
        w = self.fc2(self.fc1(w))
        return x * tf.expand_dims(w, 1)

def build_backbone():
    i = layers.Input((MAX_LEN,12))
    x = LeadAttention()(i)

    x = layers.Conv1D(64,15,padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual(x,64)
    x = residual(x,128,2)
    x = residual(x,256,2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    return models.Model(i,x)

backbone = build_backbone()

out = layers.Dense(1, activation="sigmoid", dtype="float32")(backbone.output)
model = models.Model(backbone.input, out)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(LR, weight_decay=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")]
)

# ---------------- TRAIN ----------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        EpochIndicator(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=4, mode="max", verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, mode="max", restore_best_weights=True
        ),
    ],
    verbose=0   # 🔴 IMPORTANT: disable Keras spam
)

# ---------------- FEATURE-LEVEL ENSEMBLE (LEGAL) ----------------
def extract_feats(ds):
    X, y = [], []
    for a,b in ds:
        X.append(backbone(a, training=False).numpy())
        y.append(b.numpy())
    return np.vstack(X), np.concatenate(y)

Xtr, ytr = extract_feats(train_ds)
Xte, yte = extract_feats(test_ds)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, ytr)

probs = clf.predict_proba(Xte)[:,1]

# ---------------- METRICS ----------------
print("\n===== FINAL RESULTS =====")
print("AUC :", round(roc_auc_score(yte, probs),4))
print("ACC :", round(accuracy_score(yte, probs>0.5),4))
