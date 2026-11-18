"""
train_vision.py
---------------
Train a Convolutional Neural Network (CNN) to classify fire vs. no-fire images.

Architecture
------------
Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool
-> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)

Outputs
-------
- models/fire_cnn_model.h5 : trained Keras CNN model
"""

import os
import sys
import logging
import numpy as np

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "images")
FIRE_DIR = os.path.join(DATA_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATA_DIR, "no_fire")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _count_images(directory):
    """Count image files in a directory (common extensions)."""
    if not os.path.isdir(directory):
        return 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_exts
    )


def build_cnn(input_shape=(224, 224, 3)):
    """Build the fire-detection CNN architecture."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    )

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    logger.info("[INFO] Starting vision model training pipeline")

    # Lazy import so the script fails fast on path issues before loading TF
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping
        logger.info("[OK] TensorFlow %s loaded", tf.__version__)
    except ImportError:
        logger.error("[WARN] TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Validate data directories
    # ------------------------------------------------------------------
    for label, d in [("fire", FIRE_DIR), ("no_fire", NO_FIRE_DIR)]:
        if not os.path.isdir(d):
            logger.error("[WARN] Image directory missing: %s", d)
            sys.exit(1)

    n_fire = _count_images(FIRE_DIR)
    n_no_fire = _count_images(NO_FIRE_DIR)
    total = n_fire + n_no_fire
    logger.info("[INFO] Images found -- fire: %d, no_fire: %d, total: %d",
                n_fire, n_no_fire, total)

    if total == 0:
        logger.error("[WARN] No images found in %s", DATA_DIR)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Data generators with augmentation
    # ------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        shear_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    logger.info("[OK] Training samples  : %d", train_generator.samples)
    logger.info("[OK] Validation samples: %d", val_generator.samples)
    logger.info("[INFO] Class indices    : %s", train_generator.class_indices)

    # ------------------------------------------------------------------
    # 3. Build CNN
    # ------------------------------------------------------------------
    model = build_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # 4. Callbacks
    # ------------------------------------------------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("[INFO] Starting training for %d epochs ...", EPOCHS)
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop],
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)

    print("\n" + "=" * 60)
    print("  VISION MODEL -- EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Validation Accuracy : {val_acc:.4f}")
    print(f"  Validation Loss     : {val_loss:.4f}")
    print("=" * 60)

    # Training history summary
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n  Best epoch (min val_loss): {best_epoch}")
    print(f"  Best validation accuracy : {best_val_acc:.4f}")
    print()

    # Plot suggestion
    print("  [INFO] To visualize training curves, run:")
    print("         import matplotlib.pyplot as plt")
    print("         plt.plot(history.history['accuracy'], label='train_acc')")
    print("         plt.plot(history.history['val_accuracy'], label='val_acc')")
    print("         plt.legend(); plt.title('Accuracy'); plt.show()")
    print()
    print("         plt.plot(history.history['loss'], label='train_loss')")
    print("         plt.plot(history.history['val_loss'], label='val_loss')")
    print("         plt.legend(); plt.title('Loss'); plt.show()")
    print()

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "fire_cnn_model.h5")
    model.save(model_path)
    logger.info("[OK] Saved CNN model -> %s", model_path)
    logger.info("[OK] Vision training pipeline finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("[WARN] Training failed with error: %s", exc)
        sys.exit(1)
