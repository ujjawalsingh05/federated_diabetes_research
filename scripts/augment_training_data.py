import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

print("Loading original UCI dataset...")
X_orig = pd.read_csv(RAW_DIR / "features.csv")
y_orig = pd.read_csv(RAW_DIR / "targets.csv")

# Recreating the exact 70/30 stratified split from your baseline
X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y_orig, test_size=0.3, random_state=42, stratify=y_orig
)

print(f"Original training records: {len(X_train)}")
print(f"Original positive training records: {int(y_train.sum().iloc[0])}")

print("\nLoading new BRFSS positive records...")
df_brfss = pd.read_csv(PROC_DIR / "brfss_positive_records.csv")

# Separate features and target, enforcing the exact column order of the UCI dataset
X_brfss = df_brfss[X_train.columns]
y_brfss = df_brfss[['Diabetes_binary']]

print(f"New positive records to inject: {len(X_brfss)}")

# Append the new records to the training set ONLY
X_train_aug = pd.concat([X_train, X_brfss], ignore_index=True)
y_train_aug = pd.concat([y_train, y_brfss], ignore_index=True)

# Shuffle the augmented dataset to prevent sequential bias during DNN training
X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

print(f"\nAugmented training records: {len(X_train_aug)}")
print(f"Augmented positive training records: {int(y_train_aug.sum().iloc[0])}")
print(f"Untouched test records: {len(X_test)} (Baseline integrity preserved)")

print("\nSaving augmented datasets to data/processed/ ...")
X_train_aug.to_csv(PROC_DIR / "X_train_aug.csv", index=False)
y_train_aug.to_csv(PROC_DIR / "y_train_aug.csv", index=False)
X_test.to_csv(PROC_DIR / "X_test.csv", index=False)
y_test.to_csv(PROC_DIR / "y_test.csv", index=False)

print("Data augmentation complete.")