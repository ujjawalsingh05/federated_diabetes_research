
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "prepared_data.npz"
MODEL_FILE = ROOT / "models" / "centralized_dnn.keras"
OUT = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Prepared data or trained model is missing."
        )

    data = np.load(DATA_FILE)
    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)

    model = tf.keras.models.load_model(MODEL_FILE)

    start = time.perf_counter()
    probabilities = model.predict(X_test, verbose=0).ravel()
    inference_seconds = time.perf_counter() - start

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_class_1": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_class_1": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_class_1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro",
                     zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted",
                     zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold": 0.5,
        "test_rows": int(len(y_test)),
        "total_inference_seconds": round(inference_seconds, 4),
        "mean_inference_seconds_per_sample": (
            inference_seconds / len(y_test)
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No diabetes (0)", "Diabetes (1)"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    (OUT / "centralized_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT / "centralized_classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(OUT / "confusion_matrix.csv")

    print("\nFINAL TEST METRICS")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No diabetes (0)", "Diabetes (1)"],
        zero_division=0,
    ))

    print("\nCONFUSION MATRIX")
    print(matrix)
    print("\nResults saved in:", OUT)


if __name__ == "__main__":
    main()
