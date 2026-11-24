import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
CLIENTS_DIR = ROOT / "data" / "clients" / "augmented"

# Set up argument parser so we can use the same script for both IID and Non-IID
parser = argparse.ArgumentParser(description="Run FedAvg Simulation")
parser.add_argument('--dist', type=str, default='iid', choices=['iid', 'non_iid'], help="Data partition type")
args = parser.parse_args()

DISTRIBUTION = args.dist
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64

print(f"=== Starting FedAvg Simulation ({DISTRIBUTION.upper()} Data) ===")

# 1. Load global scaler and test set
print("Loading test set and fitting global scaler...")
X_train_global = pd.read_csv(PROC_DIR / "X_train_aug.csv")
scaler = StandardScaler()
scaler.fit(X_train_global) 

X_test = pd.read_csv(PROC_DIR / "X_test.csv")
y_test = pd.read_csv(PROC_DIR / "y_test.csv")
X_test_scaled = scaler.transform(X_test)

# 2. Define the Neural Network (Strictly identical to Centralized Baseline)
def build_model():
    model = Sequential([
        Input(shape=(21,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

global_model = build_model()
global_weights = global_model.get_weights()

# 3. Load Client Data into Memory
print("Loading client data partitions...")
client_data = []
for i in range(NUM_CLIENTS):
    c_dir = CLIENTS_DIR / DISTRIBUTION / f"client_{i:02d}"
    cX = pd.read_csv(c_dir / "X_train.csv")
    cy = pd.read_csv(c_dir / "y_train.csv")
    cX_scaled = scaler.transform(cX)
    client_data.append((cX_scaled, cy, len(cX)))

total_samples = sum([c[2] for c in client_data])
print(f"Total training samples across {NUM_CLIENTS} clients: {total_samples}\n")

# 4. Federated Training Loop
start_time = time.time()

for round_num in range(1, ROUNDS + 1):
    print(f"--- Communication Round {round_num}/{ROUNDS} ---")
    client_weights = []
    
    # Simulate parallel local training
    for i in range(NUM_CLIENTS):
        X_local, y_local, n_samples = client_data[i]
        
        local_model = build_model()
        local_model.set_weights(global_weights) # Download global model
        
        # Train locally
        local_model.fit(X_local, y_local, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        
        # Upload updated weights
        client_weights.append((local_model.get_weights(), n_samples))
        print(f"  Client {i:02d} updated weights based on {n_samples} samples.")

    # FedAvg Aggregation on the Server
    new_global_weights = []
    for weight_idx in range(len(global_weights)):
        layer_weights = np.zeros_like(global_weights[weight_idx])
        for weights, n_samples in client_weights:
            layer_weights += weights[weight_idx] * (n_samples / total_samples)
        new_global_weights.append(layer_weights)
        
    global_weights = new_global_weights
    global_model.set_weights(global_weights)
    
    # Evaluate current global model
    y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f">> Global Model Validation -> Accuracy: {acc*100:.2f}%, F1-score: {f1*100:.2f}%\n")

total_time = time.time() - start_time

# 5. Final Metrics Logging
print("==============================================")
print(f"  FEDAVG {DISTRIBUTION.upper()} FINAL EVALUATION RESULTS")
print("==============================================")
y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"Total FedAvg Simulation Time: {total_time:.2f} seconds")

cm = confusion_matrix(y_test, y_pred)
print("==============================================")
print("Confusion Matrix:")
print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  | True Positives:  {cm[1][1]}")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / f"fedavg_{DISTRIBUTION}_dnn.keras"
global_model.save(model_path)
print(f"\nSaved global FedAvg {DISTRIBUTION.upper()} model to {model_path}")