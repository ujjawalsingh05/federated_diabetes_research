
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diabetes_ml.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "prepared_data.npz"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "centralized_dnn.keras"
HISTORY_FILE = METRICS_DIR / "training_history.csv"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run scripts/prepare_data.py before training."
        )

    if MODEL_FILE.exists():
        raise FileExistsError(
            "A trained model already exists. Back it up before retraining."
        )

    tf.keras.utils.set_random_seed(42)

    data = np.load(DATA_FILE)

    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Class weights compensate for the imbalanced target.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    duration = time.perf_counter() - start

    model.save(MODEL_FILE)
    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False)

    report = {
        "random_seed": 42,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(
            np.argmin(history.history["val_loss"]) + 1
        ),
        "training_duration_seconds": round(duration, 2),
        "class_weights": class_weights,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_used_during_training": False,
    }

    (METRICS_DIR / "training_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nTraining complete.")
    print("Duration (seconds):", round(duration, 2))
    print("Best epoch:", report["best_epoch"])
    print("Saved model:", MODEL_FILE)
    print("Saved history:", HISTORY_FILE)


if __name__ == "__main__":
    main()
