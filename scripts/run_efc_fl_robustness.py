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
import random

# 1. Project Paths
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
CLIENTS_DIR = ROOT / "data" / "clients" / "augmented"

parser = argparse.ArgumentParser(description="Run EFC-FL with System Heterogeneity (Dropout)")
parser.add_argument('--dist', type=str, default='non_iid', choices=['iid', 'non_iid'])
parser.add_argument('--dropout', type=float, default=0.20, help="Fraction of clients that drop out per round")
args = parser.parse_args()

DISTRIBUTION = args.dist
DROPOUT_RATE = args.dropout
MU = 0.01
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 2. Topology
EDGE_TOPOLOGY = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}
FOG_TOPOLOGY = {0: [0, 1, 2], 1: [3, 4]}

print(f"=== Starting EFC-FL Robustness Simulation ===")
print(f"Distribution: {DISTRIBUTION.upper()} | Client Dropout Rate: {DROPOUT_RATE*100:.0f}%\n")

# 3. Load Data & Scaler
X_train_global = pd.read_csv(PROC_DIR / "X_train_aug.csv")
scaler = StandardScaler().fit(X_train_global) 
X_test = pd.read_csv(PROC_DIR / "X_test.csv")
y_test = pd.read_csv(PROC_DIR / "y_test.csv")
X_test_scaled = scaler.transform(X_test)

# 4. Model Setup
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
global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy')
global_weights = global_model.get_weights()

# 5. Load Clients
client_data = {}
client_samples = {}
for i in range(10):
    c_dir = CLIENTS_DIR / DISTRIBUTION / f"client_{i:02d}"
    cX = pd.read_csv(c_dir / "X_train.csv")
    cy = pd.read_csv(c_dir / "y_train.csv")
    client_data[i] = (scaler.transform(cX), cy.values.astype(np.float32))
    client_samples[i] = len(cX)

# 6. Training Loop with Dropout
start_time = time.time()

for round_num in range(1, ROUNDS + 1):
    print(f"--- Round {round_num}/{ROUNDS} ---")
    
    # Determine which clients stay online this round
    num_active = max(1, int(10 * (1 - DROPOUT_RATE)))
    active_clients = set(random.sample(range(10), num_active))
    print(f"  Active Clients: {list(active_clients)} (Dropped: {10 - num_active})")
    
    client_deltas = {}
    
    # LEVEL 1: CLIENT TRAINING
    for i in active_clients:
        X_local, y_local = client_data[i]
        local_model = build_model()
        local_model.set_weights(global_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        dataset = tf.data.Dataset.from_tensor_slices((X_local, y_local)).shuffle(1024).batch(BATCH_SIZE)
        
        for epoch in range(LOCAL_EPOCHS):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    y_pred = local_model(x_batch, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, y_pred))
                    proximal_term = sum(tf.reduce_sum(tf.square(w_local - w_glob)) for w_local, w_glob in zip(local_model.trainable_weights, global_model.trainable_weights))
                    loss += (MU / 2.0) * proximal_term
                grads = tape.gradient(loss, local_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
        
        client_deltas[i] = [lw - gw for lw, gw in zip(local_model.get_weights(), global_weights)]
        print(f"  Client {i:02d} trained successfully.")

    # LEVEL 2: EDGE AGGREGATION (Only from active clients)
    edge_deltas = {}
    edge_samples = {}
    for edge_id, clients in EDGE_TOPOLOGY.items():
        active_edge_clients = [c for c in clients if c in active_clients]
        if not active_edge_clients:
            continue # Edge node drops out if all its clients drop out
            
        e_samples = sum(client_samples[c] for c in active_edge_clients)
        edge_samples[edge_id] = e_samples
        e_delta = [np.zeros_like(w) for w in global_weights]
        for c in active_edge_clients:
            weight_ratio = client_samples[c] / e_samples
            e_delta = [ed + (cd * weight_ratio) for ed, cd in zip(e_delta, client_deltas[c])]
        edge_deltas[edge_id] = e_delta

    # LEVEL 3: FOG AGGREGATION
    fog_deltas = {}
    fog_samples = {}
    for fog_id, edges in FOG_TOPOLOGY.items():
        active_fog_edges = [e for e in edges if e in edge_deltas]
        if not active_fog_edges:
            continue
            
        f_samples = sum(edge_samples[e] for e in active_fog_edges)
        fog_samples[fog_id] = f_samples
        f_delta = [np.zeros_like(w) for w in global_weights]
        for e in active_fog_edges:
            weight_ratio = edge_samples[e] / f_samples
            f_delta = [fd + (ed * weight_ratio) for fd, ed in zip(f_delta, edge_deltas[e])]
        fog_deltas[fog_id] = f_delta

    # LEVEL 4: CLOUD GLOBAL SYNCHRONIZATION
    global_delta = [np.zeros_like(w) for w in global_weights]
    total_active_samples = sum(fog_samples.values())
    
    if total_active_samples > 0:
        for fog_id, f_delta in fog_deltas.items():
            weight_ratio = fog_samples[fog_id] / total_active_samples
            global_delta = [gd + (fd * weight_ratio) for gd, fd in zip(global_delta, f_delta)]
        
        global_weights = [gw + gd for gw, gd in zip(global_weights, global_delta)]
        global_model.set_weights(global_weights)

total_time = time.time() - start_time

# 7. Final Evaluation
print("\n==============================================")
print(f"  EFC-FL DROPOUT (20%) {DISTRIBUTION.upper()} RESULTS")
print("==============================================")
y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_prob)*100:.2f}%")
print(f"Total Simulation Time: {total_time:.2f} seconds")