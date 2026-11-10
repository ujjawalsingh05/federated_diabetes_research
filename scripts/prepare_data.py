
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

PROCESSED.mkdir(parents=True, exist_ok=True)

SEED = 42


def main():
    output_file = PROCESSED / "prepared_data.npz"
    preprocessor_file = PROCESSED / "preprocessor.joblib"
    split_file = METADATA / "split_metadata.json"

    if output_file.exists() or preprocessor_file.exists():
        raise FileExistsError(
            "Prepared data already exists. Back it up or move it "
            "before rerunning this script."
        )

    X = pd.read_csv(RAW / "features.csv")
    y_df = pd.read_csv(RAW / "targets.csv")

    if y_df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, found {y_df.columns.tolist()}"
        )

    if X.empty or len(X) != len(y_df):
        raise ValueError("Features and targets are empty or mismatched.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature column names detected.")

    y = y_df.iloc[:, 0]

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")

    # This dataset should contain numeric health indicators.
    X = X.apply(pd.to_numeric, errors="raise")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Split BEFORE fitting any preprocessing steps.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    # Split the remaining 30% equally into validation and test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    # Learn imputation and scaling parameters from training data only.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Ensure every processed feature is finite.
    for name, values in [
        ("training", X_train_processed),
        ("validation", X_val_processed),
        ("test", X_test_processed),
    ]:
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in {name} data.")

    # Save prepared arrays without changing the original raw CSV files.
    np.savez_compressed(
        output_file,
        X_train=X_train_processed.astype(np.float32),
        X_val=X_val_processed.astype(np.float32),
        X_test=X_test_processed.astype(np.float32),
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=np.array(X.columns),
        target_classes=np.array(sorted(y.unique())),
    )

    joblib.dump(preprocessor, preprocessor_file)

    split_info = {
        "dataset_id": 891,
        "seed": SEED,
        "target_column": y_df.columns[0],
        "feature_count": int(X.shape[1]),
        "class_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "partitions": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "preprocessing": [
            "Median imputation fitted on training data only",
            "StandardScaler fitted on training data only",
        ],
    }

    split_file.write_text(json.dumps(split_info, indent=2))

    print("Data preparation complete.")
    print("Training:", X_train_processed.shape, y_train.shape)
    print("Validation:", X_val_processed.shape, y_val.shape)
    print("Test:", X_test_processed.shape, y_test.shape)
    print("\nClass distribution:")
    print(y.value_counts().sort_index())
    print("\nSaved:", output_file)
    print("Saved:", preprocessor_file)
    print("Saved:", split_file)


if __name__ == "__main__":
    main()
