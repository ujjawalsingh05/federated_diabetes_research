
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_dim: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Build a DNN for binary diabetes-related classification."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model
