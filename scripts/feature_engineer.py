import pandas as pd
import numpy as np
import os

# Paths
DATASET_DIR = "Dataset"
HANDLED_PATH = os.path.join(DATASET_DIR, "neo_model.csv")
PROCESSED_PATH = os.path.join(DATASET_DIR, "neo_processed.csv")

# Ensure folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(HANDLED_PATH)


# ---- Step 2: Feature Engineering ----
df_proc = df.copy()

# Hazard label
df_proc['hazardous_label'] = ((df_proc['moid'] < 0.05) & (df_proc['H'] <= 22)).astype(int)

# Risk score (avoid divide-by-zero)
df_proc['risk_score'] = 1 / (df_proc['moid'].replace(0, np.nan) * df_proc['H'].replace(0, np.nan))

# Perihelion ratio
df_proc['perihelion_ratio'] = df_proc['q'] / df_proc['a']

# Eccentric energy
df_proc['eccentric_energy'] = (df_proc['e'] ** 2) * df_proc['a']

# Uncertainty total (mean of sigma_* columns)
sigma_cols = [c for c in df_proc.columns if c.startswith('sigma_')]
if sigma_cols:
    df_proc['uncertainty_total'] = df_proc[sigma_cols].mean(axis=1)

# Observation span in years
df_proc['observation_span_years'] = df_proc['data_arc'] / 365

# Save processed dataset
df_proc.to_csv(PROCESSED_PATH, index=False)

print(f"✅neo_processed.csv saved in {DATASET_DIR}/")
print(f"Final shape: {df_proc.shape}")
