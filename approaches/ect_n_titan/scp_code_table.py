# ------------------------------------------------------------
# SCP Code Lists for MI and NORM (PTB-XL)
# ------------------------------------------------------------

import os

import kagglehub
import pandas as pd

# Download PTB-XL dataset
dataset_path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")

# Locate scp_statements.csv
scp_path = None
for root, _, files in os.walk(dataset_path):
    if "scp_statements.csv" in files:
        scp_path = os.path.join(root, "scp_statements.csv")
        break

if scp_path is None:
    raise FileNotFoundError("scp_statements.csv not found inside PTB-XL dataset")

# Load SCP statements
scp = pd.read_csv(scp_path)

# First column = SCP code
scp_code_col = scp.columns[0]
scp[scp_code_col] = scp[scp_code_col].astype(str)

# Keep only diagnostic statements
scp = scp[scp["diagnostic"] == 1]

# Split MI and NORM
mi_df = scp[scp["diagnostic_class"] == "MI"][[scp_code_col, "description"]]
norm_df = scp[scp["diagnostic_class"] == "NORM"][[scp_code_col, "description"]]

# Rename for paper clarity
mi_df.columns = ["SCP_Code", "Description"]
norm_df.columns = ["SCP_Code", "Description"]

# Sort for clean tables
mi_df = mi_df.sort_values("SCP_Code").reset_index(drop=True)
norm_df = norm_df.sort_values("SCP_Code").reset_index(drop=True)

# ----------------------------
# PRINT PAPER-READY TABLES
# ----------------------------
print("\n==============================")
print("Table S1: SCP Codes for Myocardial Infarction (MI)")
print("==============================")
print(mi_df.to_string(index=False))

print("\n==============================")
print("Table S2: SCP Codes for Normal ECG (NORM)")
print("==============================")
print(norm_df.to_string(index=False))

# ----------------------------
# LaTeX (for journal / conference)
# ----------------------------
print("\n% ===== LaTeX: Supplementary Tables =====")

print("\n==============================")
print("Table S1: SCP Codes for Myocardial Infarction (MI)")
print("==============================")
print(mi_df.to_markdown(index=False))

print("\n==============================")
print("Table S2: SCP Codes for Normal ECG (NORM)")
print("==============================")
print(norm_df.to_markdown(index=False))
# =================================