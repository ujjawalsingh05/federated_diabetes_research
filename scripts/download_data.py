
from pathlib import Path
from datetime import datetime, timezone
import json

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"

RAW.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)


def main():
    features_file = RAW / "features.csv"
    targets_file = RAW / "targets.csv"

    if features_file.exists() or targets_file.exists():
        print("Dataset files already exist. Inspect them before downloading again.")
        return

    print("Fetching UCI dataset ID 891...")
    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features
    y = dataset.data.targets

    if X is None or y is None or X.empty or y.empty:
        raise ValueError("The dataset returned empty features or targets.")

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    X.to_csv(features_file, index=False)
    y.to_csv(targets_file, index=False)
    dataset.variables.to_csv(META / "dataset_variables.csv", index=False)

    metadata = {
        "dataset_id": 891,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_shape": list(X.shape),
        "target_shape": list(y.shape),
        "feature_columns": X.columns.tolist(),
        "target_columns": y.columns.tolist(),
        "metadata": str(dataset.metadata),
    }

    (META / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDownload complete.")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Feature columns:", X.columns.tolist())
    print("Target columns:", y.columns.tolist())
    print("\nTarget preview:")
    print(y.head())


if __name__ == "__main__":
    main()
