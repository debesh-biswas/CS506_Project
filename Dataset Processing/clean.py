import pandas as pd
import numpy as np
import os

# Paths
DATASET_DIR = "Dataset"
RAW_PATH = os.path.join(DATASET_DIR, "neo_raw.csv")
CLEAN_PATH = os.path.join(DATASET_DIR, "neo_clean.csv")


# Ensure folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_PATH)

# ---- Step 1: Drop columns ----
drop_cols = [
    'spkid','full_name','pdes','name','prefix','orbit_id','producer','equinox','epoch_mjd',
    'tp','tp_cal','first_obs','last_obs','M1','M2','K1','K2','PC','BV','UB','IR','extent',
    'H_sigma','diameter_sigma','A1','A1_sigma','A2','A2_sigma','A3','A3_sigma','DT','pha','neo','DT_sigma'
]

df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Save processed dataset
df_clean.to_csv(CLEAN_PATH, index=False)

print(f"✅neo_clean.csv saved in {DATASET_DIR}/")
print(f"Final shape: {df_clean.shape}")
