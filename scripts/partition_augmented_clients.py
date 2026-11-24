import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
CLIENTS_DIR = ROOT / "data" / "clients" / "augmented"

IID_DIR = CLIENTS_DIR / "iid"
NON_IID_DIR = CLIENTS_DIR / "non_iid"
IID_DIR.mkdir(parents=True, exist_ok=True)
NON_IID_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 10

print("Loading augmented training dataset...")
X_train = pd.read_csv(PROC_DIR / "X_train_aug.csv")
y_train = pd.read_csv(PROC_DIR / "y_train_aug.csv")

# Combine temporarily for synchronized shuffling and sorting
df = pd.concat([X_train, y_train], axis=1)
print(f"Total records to partition: {len(df)}")

# Calculate base chunk size
chunk_size = len(df) // NUM_CLIENTS

# --- 1. IID PARTITIONING (Random, Even Distribution) ---
print("\n=== Creating IID Partitions ===")
df_iid = df.sample(frac=1, random_state=42).reset_index(drop=True)

for i in range(NUM_CLIENTS):
    # Pure Pandas slicing to guarantee it remains a DataFrame
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < NUM_CLIENTS - 1 else len(df_iid)
    client_df = df_iid.iloc[start_idx:end_idx]
    
    client_dir = IID_DIR / f"client_{i:02d}"
    client_dir.mkdir(exist_ok=True)
    
    client_X = client_df.drop(columns=['Diabetes_binary'])
    client_y = client_df[['Diabetes_binary']]
    
    client_X.to_csv(client_dir / "X_train.csv", index=False)
    client_y.to_csv(client_dir / "y_train.csv", index=False)
    print(f"IID Client {i:02d}: {len(client_df)} records (Positives: {int(client_y.sum().iloc[0])})")

# --- 2. NON-IID PARTITIONING (Label Skew) ---
print("\n=== Creating Non-IID Partitions (Label Skew) ===")
# Sorting by target variable creates an extreme real-world imbalance across clinics
df_non_iid = df.sort_values(by='Diabetes_binary').reset_index(drop=True)

for i in range(NUM_CLIENTS):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < NUM_CLIENTS - 1 else len(df_non_iid)
    
    # Extract chunk, then shuffle internally
    client_df = df_non_iid.iloc[start_idx:end_idx].sample(frac=1, random_state=42).reset_index(drop=True)
    
    client_dir = NON_IID_DIR / f"client_{i:02d}"
    client_dir.mkdir(exist_ok=True)
    
    client_X = client_df.drop(columns=['Diabetes_binary'])
    client_y = client_df[['Diabetes_binary']]
    
    client_X.to_csv(client_dir / "X_train.csv", index=False)
    client_y.to_csv(client_dir / "y_train.csv", index=False)
    print(f"Non-IID Client {i:02d}: {len(client_df)} records (Positives: {int(client_y.sum().iloc[0])})")

print("\nPartitioning complete! Data is ready for Federated Learning.")