"""
tests/test_layer1.py
Unit tests for Layer 1: Weather Risk Module
"""
import os
import sys
import pickle
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_model_dir(tmp_path):
    """Creates a temporary models directory with minimal valid model artifacts."""
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    days = ['mon','tue','wed','thu','fri','sat','sun']

    le_month = LabelEncoder().fit(months)
    le_day = LabelEncoder().fit(days)

    # Train a tiny RF on random data
    np.random.seed(0)
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)
    clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)

    with open(tmp_path / "weather_risk_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(tmp_path / "le_month.pkl", "wb") as f:
        pickle.dump(le_month, f)
    with open(tmp_path / "le_day.pkl", "wb") as f:
        pickle.dump(le_day, f)

    return str(tmp_path)


@pytest.fixture
def valid_weather_input():
    return {
        "month": "jun", "day": "fri",
        "FFMC": 86.0, "DMC": 100.0, "DC": 400.0,
        "ISI": 10.0, "temp": 30.0, "RH": 35.0,
        "wind": 5.0, "rain": 0.0
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestWeatherRiskPredictor:

    def test_loads_successfully(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        assert predictor.model is not None
        assert predictor.le_month is not None
        assert predictor.le_day is not None

    def test_raises_on_missing_models(self, tmp_path):
        from layer1_weather.predict import WeatherRiskPredictor, WeatherModelNotFoundError
        with pytest.raises(WeatherModelNotFoundError):
            WeatherRiskPredictor(models_dir=str(tmp_path))

    def test_predict_returns_correct_schema(self, dummy_model_dir, valid_weather_input):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict(valid_weather_input)
        assert "risk_level" in result
        assert "risk_score" in result
        assert result["risk_level"] in ("Low", "Medium", "High")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_predict_with_missing_fields_uses_defaults(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        # Pass empty dict — all fields missing
        result = predictor.predict({})
        assert "risk_level" in result
        assert "risk_score" in result

    def test_predict_with_partial_missing_fields(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict({"temp": 35.0, "RH": 20.0})
        assert "risk_level" in result

    def test_predict_with_invalid_type_falls_back(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict({"month": "jun", "day": "fri", "temp": "hot"})
        assert "risk_level" in result  # Should not crash

    def test_unknown_month_uses_default(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict({"month": "xyz", "day": "fri"})
        assert "risk_level" in result

    def test_unknown_day_uses_default(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict({"month": "jan", "day": "zzz"})
        assert "risk_level" in result

    def test_negative_values_clamped(self, dummy_model_dir):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict({"FFMC": -50.0, "DMC": -10.0})
        assert "risk_level" in result

    def test_risk_score_in_valid_range(self, dummy_model_dir, valid_weather_input):
        from layer1_weather.predict import WeatherRiskPredictor
        predictor = WeatherRiskPredictor(models_dir=dummy_model_dir)
        result = predictor.predict(valid_weather_input)
        assert 0.0 <= result["risk_score"] <= 1.0
