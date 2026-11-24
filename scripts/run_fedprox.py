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

parser = argparse.ArgumentParser(description="Run FedProx Simulation")
parser.add_argument('--dist', type=str, default='iid', choices=['iid', 'non_iid'], help="Data partition type")
parser.add_argument('--mu', type=float, default=0.01, help="FedProx proximal term coefficient")
args = parser.parse_args()

DISTRIBUTION = args.dist
MU = args.mu
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

print(f"=== Starting FedProx Simulation ({DISTRIBUTION.upper()} Data, Mu={MU}) ===")

print("Loading test set and fitting global scaler...")
X_train_global = pd.read_csv(PROC_DIR / "X_train_aug.csv")
scaler = StandardScaler()
scaler.fit(X_train_global) 

X_test = pd.read_csv(PROC_DIR / "X_test.csv")
y_test = pd.read_csv(PROC_DIR / "y_test.csv")
X_test_scaled = scaler.transform(X_test)

def build_model():
    model = Sequential([
        Input(shape=(21,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

global_model = build_model()
global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
global_weights = global_model.get_weights()

print("Loading client data partitions...")
client_data = []
for i in range(NUM_CLIENTS):
    c_dir = CLIENTS_DIR / DISTRIBUTION / f"client_{i:02d}"
    cX = pd.read_csv(c_dir / "X_train.csv")
    cy = pd.read_csv(c_dir / "y_train.csv")
    cX_scaled = scaler.transform(cX)
    client_data.append((cX_scaled, cy.values.astype(np.float32), len(cX)))

total_samples = sum([c[2] for c in client_data])
print(f"Total training samples across {NUM_CLIENTS} clients: {total_samples}\n")

start_time = time.time()

for round_num in range(1, ROUNDS + 1):
    print(f"--- Communication Round {round_num}/{ROUNDS} ---")
    client_weights = []
    
    global_model.set_weights(global_weights)
    
    for i in range(NUM_CLIENTS):
        X_local, y_local, n_samples = client_data[i]
        
        local_model = build_model()
        local_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        local_model.set_weights(global_weights)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        dataset = tf.data.Dataset.from_tensor_slices((X_local, y_local)).shuffle(buffer_size=1024).batch(BATCH_SIZE)
        
        for epoch in range(LOCAL_EPOCHS):
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    y_pred = local_model(x_batch, training=True)
                    loss_value = tf.keras.losses.binary_crossentropy(y_batch, y_pred)
                    base_loss = tf.reduce_mean(loss_value)
                    
                    proximal_term = 0.0
                    for w_local, w_glob in zip(local_model.trainable_weights, global_model.trainable_weights):
                        proximal_term += tf.reduce_sum(tf.square(w_local - w_glob))
                    
                    loss = base_loss + (MU / 2.0) * proximal_term
                
                grads = tape.gradient(loss, local_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
                
        client_weights.append((local_model.get_weights(), n_samples))
        print(f"  Client {i:02d} trained with FedProx (Mu={MU}).")

    new_global_weights = []
    for weight_idx in range(len(global_weights)):
        layer_weights = np.zeros_like(global_weights[weight_idx])
        for weights, n_samples in client_weights:
            layer_weights += weights[weight_idx] * (n_samples / total_samples)
        new_global_weights.append(layer_weights)
        
    global_weights = new_global_weights
    global_model.set_weights(global_weights)
    
    y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f">> Global Model Validation -> Accuracy: {acc*100:.2f}%, F1-score: {f1*100:.2f}%\n")

total_time = time.time() - start_time

print("==============================================")
print(f"  FEDPROX {DISTRIBUTION.upper()} FINAL EVALUATION RESULTS")
print("==============================================")
y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"Total FedProx Simulation Time: {total_time:.2f} seconds")

cm = confusion_matrix(y_test, y_pred)
print("==============================================")
print("Confusion Matrix:")
print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  | True Positives:  {cm[1][1]}")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / f"fedprox_{DISTRIBUTION}_dnn.keras"
global_model.save(model_path)
print(f"\nSaved global FedProx {DISTRIBUTION.upper()} model to {model_path}")