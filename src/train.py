
"""Train and evaluate ResNet50 on APTOS binary dataset."""

import json
from pathlib import Path

import tensorflow as tf

from src.data import load_data, CLASS_NAMES
from src.model import build_resnet50


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet50_aptos.keras"

EPOCHS = 15


def train_model():

    # Load dataset
    train_ds, val_ds, test_ds = load_data()

    # Build model
    model = build_resnet50()

    model.summary()

    # Callbacks
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Load best saved model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_ds)

    print("\nTest Results")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Save class names
    with open(MODEL_DIR / "class_names.json", "w") as file:
        json.dump(CLASS_NAMES, file, indent=4)

    print("\nModel saved at:", MODEL_PATH)

    return history, model


if __name__ == "__main__":
    train_model()
