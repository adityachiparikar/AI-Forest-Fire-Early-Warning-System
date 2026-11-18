import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Generating dummy models in:", models_dir)
    
    # 1. Weather Encoders & Classifier
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    
    le_month = LabelEncoder()
    le_month.fit(months)
    with open(os.path.join(models_dir, "le_month.pkl"), "wb") as f:
        pickle.dump(le_month, f)
    
    le_day = LabelEncoder()
    le_day.fit(days)
    with open(os.path.join(models_dir, "le_day.pkl"), "wb") as f:
        pickle.dump(le_day, f)
        
    # Generate some dummy weather training data
    # Columns: month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'month': np.random.choice(months, n_samples),
        'day': np.random.choice(days, n_samples),
        'FFMC': np.random.uniform(50.0, 96.0, n_samples),
        'DMC': np.random.uniform(10.0, 290.0, n_samples),
        'DC': np.random.uniform(15.0, 800.0, n_samples),
        'ISI': np.random.uniform(0.0, 50.0, n_samples),
        'temp': np.random.uniform(5.0, 42.0, n_samples),
        'RH': np.random.uniform(15.0, 100.0, n_samples),
        'wind': np.random.uniform(1.0, 10.0, n_samples),
        'rain': np.random.uniform(0.0, 6.4, n_samples),
    })
    # Target fire variable: correlated with temp and low humidity
    df['fire'] = ((df['temp'] > 28.0) & (df['RH'] < 40.0) | (df['FFMC'] > 85.0)).astype(int)
    
    # Encode strings
    df_encoded = df.copy()
    df_encoded['month'] = le_month.transform(df['month'])
    df_encoded['day'] = le_day.transform(df['day'])
    
    X = df_encoded.drop(columns=['fire'])
    y = df_encoded['fire']
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    with open(os.path.join(models_dir, "weather_risk_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print("[OK] Saved weather_risk_model.pkl, le_month.pkl, le_day.pkl")
    
    # 2. Vision Model: CNN in Keras
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Build a tiny CNN model that accepts (224, 224, 3) and outputs (1,)
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(8, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(4, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Save model
        model_path = os.path.join(models_dir, "fire_cnn_model.h5")
        model.save(model_path)
        print(f"[OK] Saved Keras model to {model_path}")
    except ImportError:
        print("[WARN] TensorFlow not installed. Skipping Keras fire_cnn_model.h5 creation.")
        print("  (A stub model will be generated during application load if tensorflow is missing)")
        
    # 3. IoT Anomaly Model (IsolationForest or simple threshold model saved as h5/pkl)
    try:
        from sklearn.ensemble import IsolationForest
        # Normal data: temp, humidity, smoke
        normal_data = np.random.normal(loc=[25.0, 60.0, 150.0], scale=[3.0, 5.0, 20.0], size=(100, 3))
        # Add a few anomalies just to make it fit
        anom_model = IsolationForest(contamination=0.05, random_state=42)
        anom_model.fit(normal_data)
        with open(os.path.join(models_dir, "iot_anomaly_model.pkl"), "wb") as f:
            pickle.dump(anom_model, f)
        print("[OK] Saved iot_anomaly_model.pkl")
    except Exception as e:
        print(f"[WARN] Could not save IoT model: {e}")

if __name__ == "__main__":
    main()
