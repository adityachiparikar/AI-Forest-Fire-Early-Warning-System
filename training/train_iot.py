"""
train_iot.py
------------
Train anomaly-detection models on normal IoT sensor readings
(temperature, humidity, smoke) for fire detection.

Two models are produced:
1. IsolationForest  (scikit-learn)  -> models/iot_anomaly_model.pkl
2. Autoencoder      (Keras)         -> models/iot_anomaly_model.h5

Outputs
-------
- models/iot_anomaly_model.pkl  : IsolationForest model
- models/iot_anomaly_model.h5   : Keras Autoencoder model
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "normal_sensor_readings.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_isolation_forest(X, feature_cols):
    """Train an IsolationForest model and save it."""
    logger.info("[INFO] Training IsolationForest (contamination=0.05) ...")

    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso_model.fit(X)

    # Predict on training data to get baseline stats
    preds = iso_model.predict(X)
    n_normal = int((preds == 1).sum())
    n_anomaly = int((preds == -1).sum())

    logger.info("[OK] IsolationForest training complete")
    logger.info("[INFO] Baseline predictions -- normal: %d, anomaly: %d", n_normal, n_anomaly)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "iot_anomaly_model.pkl")
    joblib.dump(iso_model, model_path)
    logger.info("[OK] Saved IsolationForest -> %s", model_path)

    return iso_model, n_normal, n_anomaly


def train_autoencoder(X_scaled):
    """Train a Keras Autoencoder (3 -> 8 -> 3) and save it."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.callbacks import EarlyStopping
        logger.info("[OK] TensorFlow %s loaded for Autoencoder", tf.__version__)
    except ImportError:
        logger.warning("[WARN] TensorFlow not installed -- skipping Autoencoder training")
        return None

    logger.info("[INFO] Building Autoencoder (3 -> 8 -> 3) ...")

    input_dim = X_scaled.shape[1]  # 3

    # Encoder
    inputs = Input(shape=(input_dim,), name="encoder_input")
    encoded = Dense(8, activation="relu", name="encoder_hidden")(inputs)

    # Decoder
    decoded = Dense(input_dim, activation="linear", name="decoder_output")(encoded)

    autoencoder = Model(inputs=inputs, outputs=decoded, name="sensor_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary(print_fn=logger.info)

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    # Train
    logger.info("[INFO] Training Autoencoder for up to 100 epochs ...")
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # Reconstruction error baseline
    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    mse_per_sample = np.mean(np.square(X_scaled - reconstructions), axis=1)
    threshold = np.percentile(mse_per_sample, 95)

    logger.info("[OK] Autoencoder training complete")
    logger.info("[INFO] Reconstruction MSE -- mean: %.6f, std: %.6f", mse_per_sample.mean(), mse_per_sample.std())
    logger.info("[INFO] Anomaly threshold (95th percentile): %.6f", threshold)

    # Save
    model_path = os.path.join(MODELS_DIR, "iot_anomaly_model.h5")
    autoencoder.save(model_path)
    logger.info("[OK] Saved Autoencoder -> %s", model_path)

    return autoencoder, mse_per_sample, threshold


def main():
    logger.info("[INFO] Starting IoT anomaly detection training pipeline")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not os.path.isfile(DATA_PATH):
        logger.error("[WARN] Dataset not found at %s", DATA_PATH)
        sys.exit(1)

    logger.info("[INFO] Loading sensor data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("[OK] Dataset loaded -- shape: %s", str(df.shape))
    logger.info("[INFO] Columns: %s", list(df.columns))

    # Validate expected columns
    expected_cols = {"temp", "humidity", "smoke"}
    missing = expected_cols - set(df.columns)
    if missing:
        logger.error("[WARN] Missing columns in dataset: %s", missing)
        sys.exit(1)

    feature_cols = ["temp", "humidity", "smoke"]
    X = df[feature_cols].values

    # ------------------------------------------------------------------
    # 2. Baseline statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  IoT SENSOR DATA -- BASELINE STATISTICS")
    print("=" * 60)
    for i, col in enumerate(feature_cols):
        col_data = X[:, i]
        print(f"\n  {col:>10s}:")
        print(f"    mean   = {col_data.mean():.4f}")
        print(f"    std    = {col_data.std():.4f}")
        print(f"    min    = {col_data.min():.4f}")
        print(f"    max    = {col_data.max():.4f}")
        print(f"    median = {np.median(col_data):.4f}")
    print("\n" + "=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 3. Train IsolationForest
    # ------------------------------------------------------------------
    iso_model, n_normal, n_anomaly = train_isolation_forest(X, feature_cols)

    print("\n" + "-" * 60)
    print("  ISOLATION FOREST RESULTS")
    print("-" * 60)
    print(f"  Normal readings  : {n_normal}")
    print(f"  Anomaly readings : {n_anomaly}")
    print(f"  Anomaly rate     : {n_anomaly / len(X) * 100:.2f}%")
    print("-" * 60 + "\n")

    # ------------------------------------------------------------------
    # 4. Scale data for Autoencoder
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for inference
    scaler_path = os.path.join(MODELS_DIR, "iot_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("[OK] Saved StandardScaler -> %s", scaler_path)

    # ------------------------------------------------------------------
    # 5. Train Autoencoder
    # ------------------------------------------------------------------
    ae_result = train_autoencoder(X_scaled)

    if ae_result is not None:
        _, mse_per_sample, threshold = ae_result
        print("\n" + "-" * 60)
        print("  AUTOENCODER RESULTS")
        print("-" * 60)
        print(f"  Mean reconstruction MSE : {mse_per_sample.mean():.6f}")
        print(f"  Std  reconstruction MSE : {mse_per_sample.std():.6f}")
        print(f"  Anomaly threshold (p95) : {threshold:.6f}")
        print("-" * 60 + "\n")

    logger.info("[OK] IoT anomaly detection training pipeline finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("[WARN] Training failed with error: %s", exc)
        sys.exit(1)
