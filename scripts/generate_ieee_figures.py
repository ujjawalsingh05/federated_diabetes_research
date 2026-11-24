import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

labels = [
    'Centralized',
    'FedAvg\n(Non-IID)',
    'FedProx\n(Non-IID)',
    'EFC-FL\n(Non-IID, K=10)',
    'EFC-FL\n(20% Dropout)'
]
accuracy = [83.84, 86.07, 86.08, 86.35, 85.24]
precision = [42.99, 0.00, 74.07, 57.38, 45.29]
recall = [48.90, 0.00, 0.19, 7.77, 28.54]
f1_score = [45.75, 0.00, 0.38, 13.69, 35.01]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy', color='#1f77b4')
ax.bar(x - 0.5 * width, precision, width, label='Precision', color='#ff7f0e')
ax.bar(x + 0.5 * width, recall, width, label='Recall', color='#2ca02c')
ax.bar(x + 1.5 * width, f1_score, width, label='F1-score', color='#d62728')

ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Classification Performance Across Configurations (Non-IID)', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(output_dir / 'performance_metrics_ieee.png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.heatmap(
    [[65493, 7], [10584, 20]],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0],
    cbar=False,
    xticklabels=['Pred 0', 'Pred 1'],
    yticklabels=['Actual 0', 'Actual 1'],
    annot_kws={"size": 13, "weight": "bold"}
)
axes[0].set_title('Flat FedProx (Non-IID, K=10)', fontweight='bold', pad=10)

sns.heatmap(
    [[64888, 612], [9780, 824]],
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1],
    cbar=False,
    xticklabels=['Pred 0', 'Pred 1'],
    yticklabels=['Actual 0', 'Actual 1'],
    annot_kws={"size": 13, "weight": "bold"}
)
axes[1].set_title('Hierarchical EFC-FL (Non-IID, K=10)', fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices_ieee.png', dpi=300)
plt.close()

k_labels = ['K = 5\n(1 Fog)', 'K = 10\n(2 Fog)', 'K = 20\n(4 Fog)']
flat_payload = [138.0, 276.0, 552.0]
efc_payload = [27.6, 55.2, 110.4]

x_k = np.arange(len(k_labels))
w_k = 0.35

fig, ax = plt.subplots(figsize=(8, 4.8))
bars1 = ax.bar(x_k - w_k / 2, flat_payload, w_k, label='Flat FedAvg / FedProx', color='#7f7f7f')
bars2 = ax.bar(x_k + w_k / 2, efc_payload, w_k, label='Hierarchical EFC-FL (-80%)', color='#1f77b4')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Cloud-Facing Payload per Round (KiB)', fontweight='bold')
ax.set_title('Cloud Bottleneck Traffic Scaling Across Client Counts', fontweight='bold', pad=12)
ax.set_xticks(x_k)
ax.set_xticklabels(k_labels, fontweight='bold')
ax.set_ylim(0, 630)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(output_dir / 'scalability_cloud_traffic_ieee.png', dpi=300)
plt.close()

print(f"All figures saved to {output_dir.resolve()}")