import os
import pickle
import logging
import numpy as np
import pandas as pd
from config import MODELS_DIR

# Set up logger
logger = logging.getLogger(__name__)

# Constants for default values (safe fallback metrics)
DEFAULT_WEATHER_DATA = {
    'month': 'jun',
    'day': 'fri',
    'FFMC': 80.0,
    'DMC': 30.0,
    'DC': 100.0,
    'ISI': 5.0,
    'temp': 25.0,
    'RH': 50.0,
    'wind': 4.0,
    'rain': 0.0
}

class WeatherModelNotFoundError(FileNotFoundError):
    """Raised when weather risk model files are not found."""
    pass

class WeatherRiskPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "weather_risk_model.pkl")
        self.le_month_path = os.path.join(models_dir, "le_month.pkl")
        self.le_day_path = os.path.join(models_dir, "le_day.pkl")
        
        self.model = None
        self.le_month = None
        self.le_day = None
        self._load_models()

    def _load_models(self):
        """Loads classifier and label encoders from models folder."""
        if not (os.path.exists(self.model_path) and 
                os.path.exists(self.le_month_path) and 
                os.path.exists(self.le_day_path)):
            raise WeatherModelNotFoundError(
                f"Weather model artifacts not found in {self.models_dir}. "
                "Please run training pipelines or generate dummy models first."
            )
            
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.le_month_path, "rb") as f:
                self.le_month = pickle.load(f)
            with open(self.le_day_path, "rb") as f:
                self.le_day = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading weather model artifacts: {e}")
            raise

    def predict(self, input_data: dict) -> dict:
        """
        Runs weather fire risk classification.
        Input: dict containing weather features.
        Output: {'risk_level': str, 'risk_score': float}
        """
        if self.model is None or self.le_month is None or self.le_day is None:
            return {"risk_level": "Medium", "risk_score": 0.5, "error": "Model not loaded"}

        # Validate and impute missing/out-of-bounds fields
        cleaned_data = {}
        for key, default_val in DEFAULT_WEATHER_DATA.items():
            val = input_data.get(key, None)
            if val is None:
                logger.warning(f"Missing weather field '{key}'. Using default: {default_val}")
                cleaned_data[key] = default_val
            else:
                # Basic range validation
                if key in ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']:
                    try:
                        f_val = float(val)
                        if f_val < 0 and key not in ['temp']:  # temp can be negative in theory
                            f_val = 0.0
                        cleaned_data[key] = f_val
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid type for weather field '{key}': {val}. Using default: {default_val}")
                        cleaned_data[key] = default_val
                else:
                    cleaned_data[key] = str(val).lower()

        try:
            # Map categorical month/day strings
            # If month/day is not in the encoder classes, fall back to first class or default
            month_val = cleaned_data['month']
            if month_val not in self.le_month.classes_:
                logger.warning(f"Month '{month_val}' not in trained classes. Mapping to default 'jun'")
                month_val = 'jun'
            encoded_month = self.le_month.transform([month_val])[0]

            day_val = cleaned_data['day']
            if day_val not in self.le_day.classes_:
                logger.warning(f"Day '{day_val}' not in trained classes. Mapping to default 'fri'")
                day_val = 'fri'
            encoded_day = self.le_day.transform([day_val])[0]

            # Construct DataFrame for prediction matching column order
            feature_order = ['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
            features = {
                'month': [encoded_month],
                'day': [encoded_day],
                'FFMC': [cleaned_data['FFMC']],
                'DMC': [cleaned_data['DMC']],
                'DC': [cleaned_data['DC']],
                'ISI': [cleaned_data['ISI']],
                'temp': [cleaned_data['temp']],
                'RH': [cleaned_data['RH']],
                'wind': [cleaned_data['wind']],
                'rain': [cleaned_data['rain']]
            }
            X = pd.DataFrame(features, columns=feature_order)
            
            # Predict risk score (probability of class 1)
            risk_score = float(self.model.predict_proba(X)[0][1])
            
            # Determine Risk Level
            if risk_score < 0.35:
                risk_level = "Low"
            elif risk_score < 0.70:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            return {
                "risk_level": risk_level,
                "risk_score": risk_score
            }
            
        except Exception as e:
            logger.error(f"Inference error in Weather risk model: {e}")
            return {
                "risk_level": "Medium",
                "risk_score": 0.5,
                "error": str(e)
            }
