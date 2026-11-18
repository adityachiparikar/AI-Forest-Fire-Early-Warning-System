import os
import time
import pickle
import logging
import pandas as pd
import numpy as np
from config import MODELS_DIR, IOT_SIM_CSV

logger = logging.getLogger(__name__)

class IoTAnomalyModelNotFoundError(FileNotFoundError):
    """Exception raised when IoT anomaly detection model is missing."""
    pass

class IoTSensorPipeline:
    """
    IEEE 2413 compliance (Sensing -> Network -> Application) represented through decoupled layers.
    IEEE 1451 compliance (Standardized Sensor Transducer interface) with uniform telemetry schema.
    """
    def __init__(self, models_dir=MODELS_DIR, csv_path=IOT_SIM_CSV):
        self.models_dir = models_dir
        self.csv_path = csv_path
        self.model_path_pkl = os.path.join(models_dir, "iot_anomaly_model.pkl")
        self.model_path_h5 = os.path.join(models_dir, "iot_anomaly_model.h5")
        
        self.anomaly_model = None
        self._load_anomaly_model()
        
    def _load_anomaly_model(self):
        """Loads Isolation Forest model (pkl) or Autoencoder model (h5)."""
        if os.path.exists(self.model_path_pkl):
            try:
                with open(self.model_path_pkl, "rb") as f:
                    self.anomaly_model = pickle.load(f)
                logger.info("✓ IoT Isolation Forest anomaly model loaded (.pkl).")
            except Exception as e:
                logger.error(f"Error loading IoT anomaly pickle: {e}")
        elif os.path.exists(self.model_path_h5):
            try:
                import tensorflow as tf
                self.anomaly_model = tf.keras.models.load_model(self.model_path_h5)
                logger.info("✓ IoT Autoencoder anomaly model loaded (.h5).")
            except Exception as e:
                logger.error(f"Error loading IoT anomaly keras model: {e}")
        else:
            logger.warning("IoT anomaly model not found. System will run in Threshold-Only fallback mode.")

    def format_sensor_reading(self, temp: float, humidity: float, smoke: float, timestamp: float = None) -> dict:
        """
        IEEE 1451 standard payload: formats raw sensor values into standard, 
        self-describing Transducer Electronic Data Sheets (TEDS) schema.
        """
        return {
            "metadata": {
                "sensor_platform": "ESP32-Sahyadri-Node",
                "standard": "IEEE-1451-TEDS",
                "timestamp": timestamp if timestamp else time.time()
            },
            "channels": {
                "temperature": {"value": float(temp), "unit": "Celsius"},
                "humidity": {"value": float(humidity), "unit": "Percent"},
                "smoke_concentration": {"value": float(smoke), "unit": "PPM"}
            }
        }

    def read_simulated_stream(self, index: int) -> dict:
        """
        Simulated Path: reads Nth row from sensor_stream.csv and formats it.
        Returns standardized sensor payload or empty dict if out of bounds.
        """
        if not os.path.exists(self.csv_path):
            logger.error(f"Simulated CSV stream not found at: {self.csv_path}")
            return {}
            
        try:
            df = pd.read_csv(self.csv_path)
            if index >= len(df):
                # Wrap around simulation
                index = index % len(df)
            
            row = df.iloc[index]
            # Maintain active timestamps relative to real current time
            # by offsetting index steps
            simulated_ts = time.time() - (len(df) - index) * 10
            
            return self.format_sensor_reading(
                temp=row['temp'],
                humidity=row['humidity'],
                smoke=row['smoke'],
                timestamp=simulated_ts
            )
        except Exception as e:
            logger.error(f"Error reading CSV simulation stream: {e}")
            return {}

    def analyze_telemetry(self, sensor_payload: dict) -> dict:
        """
        Processes standard IEEE 1451 payload.
        First applies fast interpretable thresholds, then evaluates ML anomaly model.
        Output: {"iot_confidence": float, "latest_reading": dict, "anomaly_flag": bool, "stale": bool}
        """
        if not sensor_payload or "channels" not in sensor_payload:
            return {
                "iot_confidence": 0.0,
                "latest_reading": {},
                "anomaly_flag": False,
                "stale": True,
                "error": "Empty or invalid sensor payload"
            }
            
        # 1. Check for stale data (IEEE 2413 reliability constraint)
        # If timestamp is older than 5 minutes, mark as stale
        ts = sensor_payload["metadata"]["timestamp"]
        current_time = time.time()
        stale = (current_time - ts) > 300.0  # 300 seconds
        
        channels = sensor_payload["channels"]
        temp = channels["temperature"]["value"]
        humidity = channels["humidity"]["value"]
        smoke = channels["smoke_concentration"]["value"]
        
        # 2. Apply Threshold Rules (Interpretable baseline)
        # Standard indicators of fire:
        # High heat (> 40C), dry air (< 25%), elevated smoke/particulate counts (> 250 PPM)
        temp_score = min(max((temp - 30.0) / 15.0, 0.0), 1.0) if temp > 30.0 else 0.0
        hum_score = min(max((45.0 - humidity) / 30.0, 0.0), 1.0) if humidity < 45.0 else 0.0
        smoke_score = min(max((smoke - 200.0) / 600.0, 0.0), 1.0) if smoke > 200.0 else 0.0
        
        # Simple weighted threshold confidence
        threshold_confidence = (temp_score * 0.3) + (hum_score * 0.2) + (smoke_score * 0.5)
        
        # 3. Apply ML Anomaly Model if available
        ml_confidence = 0.0
        ml_anomaly_detected = False
        
        if self.anomaly_model is not None:
            try:
                # Prepare features: shape (1, 3) representing temp, humidity, smoke
                features = np.array([[temp, humidity, smoke]])
                
                # Check model type
                if hasattr(self.anomaly_model, "predict"):
                    # IsolationForest outputs: 1 for normal, -1 for anomaly
                    prediction = self.anomaly_model.predict(features)[0]
                    ml_anomaly_detected = (prediction == -1)
                    
                    # Compute score based on decision function (negative score is anomaly)
                    if hasattr(self.anomaly_model, "decision_function"):
                        dec_score = float(self.anomaly_model.decision_function(features)[0])
                        # Normalize decision score to probability-like confidence
                        # IsolationForest decision score ranges roughly from -0.5 to 0.5
                        # A lower/more negative score indicates a stronger anomaly
                        ml_confidence = min(max(-dec_score * 2.0, 0.0), 1.0) if ml_anomaly_detected else 0.0
                else:
                    # Assume Autoencoder Keras model
                    # Predict reconstructed inputs
                    reconstruction = self.anomaly_model.predict(features, verbose=0)
                    mse = np.mean(np.square(features - reconstruction))
                    # Basic reconstruction error threshold (tuned on baseline)
                    ml_anomaly_detected = mse > 15.0 
                    ml_confidence = min(mse / 100.0, 1.0)
            except Exception as e:
                logger.error(f"Error running IoT ML inference: {e}")
                
        # Final combined assessment (ML model boosts confidence of threshold detections)
        if self.anomaly_model is not None:
            # Fuse Threshold & ML anomalies
            anomaly_flag = (threshold_confidence > 0.4) or ml_anomaly_detected
            # Weighted confidence
            iot_confidence = max(threshold_confidence, ml_confidence)
        else:
            anomaly_flag = threshold_confidence > 0.4
            iot_confidence = threshold_confidence
            
        return {
            "iot_confidence": float(iot_confidence),
            "latest_reading": {
                "temp": temp,
                "humidity": humidity,
                "smoke": smoke,
                "timestamp": ts
            },
            "anomaly_flag": bool(anomaly_flag),
            "stale": bool(stale)
        }
        
    def setup_mqtt_subscriber(self, broker_address="localhost", topic="sahyadri/iot/sensors"):
        """
        IEEE 2413 Network Layer edge gateway setup hook.
        This provides a real hardware extension path.
        """
        try:
            import paho.mqtt.client as mqtt
            
            def on_connect(client, userdata, flags, rc):
                logger.info(f"MQTT Connected with result code {rc}")
                client.subscribe(topic)
                
            def on_message(client, userdata, msg):
                try:
                    import json
                    payload = json.loads(msg.payload.decode())
                    logger.info(f"MQTT Received sensor reading: {payload}")
                    # In production, pass this payload to analyze_telemetry
                except Exception as e:
                    logger.error(f"Failed to parse MQTT message: {e}")
                    
            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            # Async non-blocking connection
            client.connect_async(broker_address, 1883, 60)
            return client
        except ImportError:
            logger.warning("paho-mqtt not installed. MQTT subscription pathway is unavailable.")
            return None
