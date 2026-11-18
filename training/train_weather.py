"""
train_weather.py
----------------
Train a RandomForestClassifier on the UCI Forest Fires dataset to predict
binary fire occurrence (fire=1 if area > 0, else 0).

Outputs
-------
- models/weather_risk_model.pkl   : trained RandomForest model
- models/le_month.pkl             : LabelEncoder for 'month'
- models/le_day.pkl               : LabelEncoder for 'day'
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# Resolve paths relative to the project root (parent of training/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "forest_fires.csv")
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


def main():
    logger.info("[INFO] Starting weather risk model training pipeline")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not os.path.isfile(DATA_PATH):
        logger.error("[WARN] Dataset not found at %s", DATA_PATH)
        sys.exit(1)

    logger.info("[INFO] Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("[OK] Dataset loaded -- shape: %s", str(df.shape))
    logger.info("[INFO] Columns: %s", list(df.columns))

    # ------------------------------------------------------------------
    # 2. Create binary target
    # ------------------------------------------------------------------
    df["fire"] = (df["area"] > 0).astype(int)
    fire_counts = df["fire"].value_counts()
    logger.info(
        "[INFO] Target distribution -- no_fire(0): %d, fire(1): %d",
        fire_counts.get(0, 0),
        fire_counts.get(1, 0),
    )

    # ------------------------------------------------------------------
    # 3. Encode categorical features
    # ------------------------------------------------------------------
    le_month = LabelEncoder()
    le_day = LabelEncoder()

    df["month"] = le_month.fit_transform(df["month"])
    df["day"] = le_day.fit_transform(df["day"])

    logger.info("[OK] Encoded 'month' classes: %s", list(le_month.classes_))
    logger.info("[OK] Encoded 'day'   classes: %s", list(le_day.classes_))

    # ------------------------------------------------------------------
    # 4. Prepare features / target
    # ------------------------------------------------------------------
    feature_cols = ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI",
                    "temp", "RH", "wind", "rain"]
    X = df[feature_cols].values
    y = df["fire"].values

    logger.info("[INFO] Feature matrix shape: %s", str(X.shape))

    # ------------------------------------------------------------------
    # 5. Train / test split (80 / 20)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "[INFO] Split -- train: %d samples, test: %d samples",
        X_train.shape[0],
        X_test.shape[0],
    )

    # ------------------------------------------------------------------
    # 6. Train RandomForestClassifier
    # ------------------------------------------------------------------
    logger.info("[INFO] Training RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info("[OK] Model training complete")

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["no_fire", "fire"])

    print("\n" + "=" * 60)
    print("  WEATHER RISK MODEL -- EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy: {acc:.4f}\n")
    print(report)
    print("=" * 60)

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Feature Importances (top 5):")
    for rank, idx in enumerate(sorted_idx[:5], start=1):
        print(f"    {rank}. {feature_cols[idx]:>8s} : {importances[idx]:.4f}")
    print()

    # ------------------------------------------------------------------
    # 8. Save artifacts
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "weather_risk_model.pkl")
    le_month_path = os.path.join(MODELS_DIR, "le_month.pkl")
    le_day_path = os.path.join(MODELS_DIR, "le_day.pkl")

    joblib.dump(clf, model_path)
    joblib.dump(le_month, le_month_path)
    joblib.dump(le_day, le_day_path)

    logger.info("[OK] Saved model        -> %s", model_path)
    logger.info("[OK] Saved le_month      -> %s", le_month_path)
    logger.info("[OK] Saved le_day        -> %s", le_day_path)
    logger.info("[OK] Weather risk training pipeline finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("[WARN] Training failed with error: %s", exc)
        sys.exit(1)
