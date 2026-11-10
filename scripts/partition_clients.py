
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "prepared_data.npz"
OUTPUT_DIR = ROOT / "data" / "clients"

NUM_CLIENTS = 10
SEED = 42

# Smaller alpha means more heterogeneous client class distributions.
DIRICHLET_ALPHA = 0.5


def save_partitions(X, y, partitions, output_dir, mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": mode,
        "num_clients": len(partitions),
        "seed": SEED,
        "clients": {},
    }

    all_indices = np.concatenate(partitions)
    expected_indices = np.arange(len(y))

    if len(all_indices) != len(y):
        raise ValueError(f"{mode}: some samples are missing or duplicated.")

    if not np.array_equal(np.sort(all_indices), expected_indices):
        raise ValueError(f"{mode}: partition integrity check failed.")

    for client_id, indices in enumerate(partitions):
        client_dir = output_dir / f"client_{client_id:02d}"
        client_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            client_dir / "train.npz",
            X=X[indices].astype(np.float32),
            y=y[indices].astype(np.int32),
        )

        labels, counts = np.unique(y[indices], return_counts=True)

        summary["clients"][f"client_{client_id:02d}"] = {
            "samples": int(len(indices)),
            "class_counts": {
                str(label): int(count)
                for label, count in zip(labels, counts)
            },
        }

    (output_dir / "partition_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n{mode.upper()} partition:")
    for client, info in summary["clients"].items():
        print(
            f"{client}: {info['samples']} records | "
            f"classes={info['class_counts']}"
        )

    print("Saved:", output_dir)


def make_iid_partitions(y, num_clients, rng):
    indices = rng.permutation(len(y))
    return [
        part.astype(int)
        for part in np.array_split(indices, num_clients)
    ]


def make_non_iid_partitions(y, num_clients, alpha, rng):
    partitions = [[] for _ in range(num_clients)]
    classes = np.unique(y)

    # Allocate each class unevenly across clients using a
    # Dirichlet distribution.
    for label in classes:
        class_indices = np.flatnonzero(y == label)
        class_indices = rng.permutation(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha, dtype=float)
        )

        # Convert proportions into integer counts that sum
        # exactly to the number of samples for this class.
        raw_counts = proportions * len(class_indices)
        counts = np.floor(raw_counts).astype(int)
        remainder = len(class_indices) - counts.sum()

        if remainder:
            order = np.argsort(-(raw_counts - counts))
            counts[order[:remainder]] += 1

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            partitions[client_id].extend(
                class_indices[start:end].tolist()
            )
            start = end

    # Shuffle each client's records.
    result = []
    for indices in partitions:
        indices = np.asarray(indices, dtype=int)
        result.append(rng.permutation(indices))

    if any(len(indices) == 0 for indices in result):
        raise ValueError("An empty client was created; change alpha or seed.")

    return result


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run scripts/prepare_data.py first."
        )

    rng_iid = np.random.default_rng(SEED)
    rng_non_iid = np.random.default_rng(SEED)

    with np.load(DATA_FILE) as data:
        # Only training data are partitioned.
        X_train = data["X_train"]
        y_train = data["y_train"].astype(int)

    print("Training records:", len(y_train))
    print("Features per record:", X_train.shape[1])
    print("Clients:", NUM_CLIENTS)
    print("Class counts:", {
        str(k): int(v)
        for k, v in zip(*np.unique(y_train, return_counts=True))
    })

    iid_partitions = make_iid_partitions(
        y_train, NUM_CLIENTS, rng_iid
    )

    non_iid_partitions = make_non_iid_partitions(
        y_train, NUM_CLIENTS, DIRICHLET_ALPHA, rng_non_iid
    )

    save_partitions(
        X_train,
        y_train,
        iid_partitions,
        OUTPUT_DIR / "iid",
        "iid",
    )

    save_partitions(
        X_train,
        y_train,
        non_iid_partitions,
        OUTPUT_DIR / "non_iid",
        "non_iid",
    )

    print("\nClient partitioning completed successfully.")
    print("Validation and test data remain unchanged.")


if __name__ == "__main__":
    main()
