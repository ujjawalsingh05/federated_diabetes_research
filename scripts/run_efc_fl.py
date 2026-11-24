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

parser = argparse.ArgumentParser(description="Run Hierarchical EFC-FL Simulation")
parser.add_argument('--dist', type=str, default='non_iid', choices=['iid', 'non_iid'])
parser.add_argument('--clients', type=int, default=5, choices=[5, 10, 20])
parser.add_argument('--mu', type=float, default=0.01)
parser.add_argument('--sparsity', type=float, default=0.0)
args = parser.parse_args()

DISTRIBUTION = args.dist
NUM_CLIENTS = args.clients
MU = args.mu
SPARSITY = args.sparsity
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

TOPOLOGIES = {
    5: (
        {0: [0, 1, 2], 1: [3, 4]},
        {0: [0, 1]}
    ),
    10: (
        {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]},
        {0: [0, 1, 2], 1: [3, 4]}
    ),
    20: (
        {
            0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9],
            5: [10, 11], 6: [12, 13], 7: [14, 15], 8: [16, 17], 9: [18, 19]
        },
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8, 9]}
    )
}

EDGE_TOPOLOGY, FOG_TOPOLOGY = TOPOLOGIES[NUM_CLIENTS]

print(f"=== Starting EFC-FL Simulation ({DISTRIBUTION.upper()} Data, K={NUM_CLIENTS}) ===")
print(f"Topology: {NUM_CLIENTS} Clients -> {len(EDGE_TOPOLOGY)} Edge Nodes -> {len(FOG_TOPOLOGY)} Fog Nodes -> 1 Cloud Server")
print(f"Communication Optimization: {SPARSITY*100:.0f}% Sparsity Masking\n")

X_train_global = pd.read_csv(PROC_DIR / "X_train_aug.csv")
scaler = StandardScaler()
scaler.fit(X_train_global)

X_test = pd.read_csv(PROC_DIR / "X_test.csv")
y_test = pd.read_csv(PROC_DIR / "y_test.csv")
X_test_scaled = scaler.transform(X_test)

def build_model():
    return Sequential([
        Input(shape=(21,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

global_model = build_model()
global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy')
global_weights = global_model.get_weights()

all_X_list = []
all_y_list = []
for i in range(10):
    c_dir = CLIENTS_DIR / DISTRIBUTION / f"client_{i:02d}"
    all_X_list.append(pd.read_csv(c_dir / "X_train.csv"))
    all_y_list.append(pd.read_csv(c_dir / "y_train.csv"))

X_full = pd.concat(all_X_list, ignore_index=True)
y_full = pd.concat(all_y_list, ignore_index=True)
X_full_scaled = scaler.transform(X_full)
y_full_arr = y_full.values.astype(np.float32)

if DISTRIBUTION == 'non_iid':
    sort_idx = np.argsort(y_full_arr.flatten(), kind='stable')
    X_full_scaled = X_full_scaled[sort_idx]
    y_full_arr = y_full_arr[sort_idx]

X_splits = np.array_split(X_full_scaled, NUM_CLIENTS)
y_splits = np.array_split(y_full_arr, NUM_CLIENTS)

client_data = {}
client_samples = {}
for i in range(NUM_CLIENTS):
    client_data[i] = (X_splits[i], y_splits[i])
    client_samples[i] = len(X_splits[i])
    print(f"  Client {i:02d}: {len(X_splits[i])} samples ({int(y_splits[i].sum())} positives)")

total_samples = sum(client_samples.values())
print(f"Total training samples: {total_samples}\n")

def sparsify_delta(delta_weights, sparsity_rate):
    if sparsity_rate == 0:
        return delta_weights
    all_weights = np.concatenate([w.flatten() for w in delta_weights])
    threshold = np.percentile(np.abs(all_weights), sparsity_rate * 100)
    sparse_delta = []
    for w in delta_weights:
        mask = (np.abs(w) >= threshold).astype(np.float32)
        sparse_delta.append(w * mask)
    return sparse_delta

start_time = time.time()

for round_num in range(1, ROUNDS + 1):
    print(f"--- EFC-FL Communication Round {round_num}/{ROUNDS} ---")
    client_deltas = {}

    for i in range(NUM_CLIENTS):
        X_local, y_local = client_data[i]
        local_model = build_model()
        local_model.set_weights(global_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        dataset = tf.data.Dataset.from_tensor_slices((X_local, y_local)).shuffle(buffer_size=1024).batch(BATCH_SIZE)

        for epoch in range(LOCAL_EPOCHS):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    y_pred = local_model(x_batch, training=True)
                    base_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, y_pred))
                    proximal_term = 0.0
                    for w_local, w_glob in zip(local_model.trainable_weights, global_model.trainable_weights):
                        proximal_term += tf.reduce_sum(tf.square(w_local - w_glob))
                    loss = base_loss + (MU / 2.0) * proximal_term
                grads = tape.gradient(loss, local_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

        local_weights = local_model.get_weights()
        delta = [lw - gw for lw, gw in zip(local_weights, global_weights)]
        client_deltas[i] = sparsify_delta(delta, SPARSITY)
        print(f"  Client {i:02d} trained & sparsified ({SPARSITY*100:.0f}% payload dropped).")

    edge_deltas = {}
    edge_samples = {}
    for edge_id, clients in EDGE_TOPOLOGY.items():
        e_samples = sum(client_samples[c] for c in clients)
        edge_samples[edge_id] = e_samples
        e_delta = [np.zeros_like(w) for w in global_weights]
        for c in clients:
            weight_ratio = client_samples[c] / e_samples
            e_delta = [ed + (cd * weight_ratio) for ed, cd in zip(e_delta, client_deltas[c])]
        edge_deltas[edge_id] = e_delta
    print("  >> Edge Layer: Aggregated client updates.")

    fog_deltas = {}
    fog_samples = {}
    for fog_id, edges in FOG_TOPOLOGY.items():
        f_samples = sum(edge_samples[e] for e in edges)
        fog_samples[fog_id] = f_samples
        f_delta = [np.zeros_like(w) for w in global_weights]
        for e in edges:
            weight_ratio = edge_samples[e] / f_samples
            f_delta = [fd + (ed * weight_ratio) for fd, ed in zip(f_delta, edge_deltas[e])]
        fog_deltas[fog_id] = f_delta
    print("  >> Fog Layer: Aggregated edge updates.")

    global_delta = [np.zeros_like(w) for w in global_weights]
    for fog_id, f_delta in fog_deltas.items():
        weight_ratio = fog_samples[fog_id] / total_samples
        global_delta = [gd + (fd * weight_ratio) for gd, fd in zip(global_delta, f_delta)]

    global_weights = [gw + gd for gw, gd in zip(global_weights, global_delta)]
    global_model.set_weights(global_weights)
    print("  >> Cloud Layer: Global model synchronized.")

    y_pred_prob = global_model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f">> Global Validation -> Accuracy: {acc*100:.2f}%, F1-score: {f1*100:.2f}%\n")

total_time = time.time() - start_time

print("==============================================")
print(f"  HIERARCHICAL EFC-FL {DISTRIBUTION.upper()} (K={NUM_CLIENTS}) FINAL RESULTS")
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
print(f"Total Simulation Time: {total_time:.2f} seconds")

cm = confusion_matrix(y_test, y_pred)
print("==============================================")
print("Confusion Matrix:")
print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}  | True Positives:  {cm[1][1]}")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / f"efc_fl_{DISTRIBUTION}_k{NUM_CLIENTS}_s{int(SPARSITY*100)}.keras"
global_model.save(model_path)
print(f"\nSaved global EFC-FL model to {model_path}")