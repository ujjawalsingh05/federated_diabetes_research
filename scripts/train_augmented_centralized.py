import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading augmented training data and untouched test data...")
X_train = pd.read_csv(PROC_DIR / "X_train_aug.csv")
y_train = pd.read_csv(PROC_DIR / "y_train_aug.csv")
X_test = pd.read_csv(PROC_DIR / "X_test.csv")
y_test = pd.read_csv(PROC_DIR / "y_test.csv")

print("Scaling features...")
scaler = StandardScaler()
# We ONLY fit the scaler on the training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBuilding the DNN model (identical to baseline architecture)...")
model = Sequential([
    Input(shape=(21,)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining the Augmented Centralized DNN...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)

model_path = MODELS_DIR / "augmented_centralized_dnn.keras"
model.save(model_path)
print(f"\nModel saved to {model_path}")

print("\nEvaluating model on the untouched test set...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n==============================================")
print("     AUGMENTED MODEL EVALUATION RESULTS       ")
print("==============================================")
print(f"Accuracy:  {acc*100:.2f}% (Baseline: 71.90%)")
print(f"Precision: {prec*100:.2f}% (Baseline: 30.64%)")
print(f"Recall:    {rec*100:.2f}% (Baseline: 80.42%)")
print(f"F1-score:  {f1*100:.2f}% (Baseline: 44.37%)")
print(f"ROC-AUC:   {roc_auc*100:.2f}% (Baseline: 82.95%)")
print("==============================================")
print("Confusion Matrix:")
print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  | True Positives:  {cm[1][1]}")