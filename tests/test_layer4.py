"""
tests/test_layer4.py
Unit tests for Layer 4: IoT Sensor Pipeline
"""
import os
import sys
import time
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_csv(tmp_path):
    """Creates a minimal simulated sensor CSV file."""
    df = pd.DataFrame({
        "temp":     [24.5, 25.0, 26.0, 30.0, 40.0, 50.0],
        "humidity": [60.0, 58.0, 55.0, 45.0, 30.0, 15.0],
        "smoke":    [140., 150., 160., 200., 400., 800.],
        "timestamp":[float(time.time() - i * 10) for i in reversed(range(6))]
    })
    csv_path = tmp_path / "sensor_stream.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def pipeline_no_model(dummy_csv, tmp_path):
    """Pipeline instance with no anomaly model (threshold-only mode)."""
    from layer4_iot.sensor_pipeline import IoTSensorPipeline
    return IoTSensorPipeline(models_dir=str(tmp_path), csv_path=dummy_csv)


@pytest.fixture
def pipeline_with_isolation_forest(dummy_csv, tmp_path):
    """Pipeline instance with a fitted IsolationForest model."""
    from sklearn.ensemble import IsolationForest
    from layer4_iot.sensor_pipeline import IoTSensorPipeline
    normal = np.random.normal(loc=[25, 60, 150], scale=[2, 4, 15], size=(100, 3))
    model = IsolationForest(contamination=0.05, random_state=42).fit(normal)
    with open(tmp_path / "iot_anomaly_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return IoTSensorPipeline(models_dir=str(tmp_path), csv_path=dummy_csv)


# ---------------------------------------------------------------------------
# Tests: IEEE 1451 sensor reading format
# ---------------------------------------------------------------------------
class TestSensorReadingFormat:

    def test_format_produces_ieee1451_schema(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0)
        assert "metadata" in reading
        assert "channels" in reading
        assert "timestamp" in reading["metadata"]
        assert "standard" in reading["metadata"]
        assert "temperature" in reading["channels"]
        assert "humidity" in reading["channels"]
        assert "smoke_concentration" in reading["channels"]

    def test_channels_have_value_and_unit(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(28.0, 55.0, 180.0)
        for ch_name, ch_data in reading["channels"].items():
            assert "value" in ch_data
            assert "unit" in ch_data

    def test_temperature_unit_is_celsius(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(30.0, 50.0, 200.0)
        assert reading["channels"]["temperature"]["unit"] == "Celsius"

    def test_timestamp_set_when_not_provided(self, pipeline_no_model):
        before = time.time()
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0)
        after = time.time()
        ts = reading["metadata"]["timestamp"]
        assert before <= ts <= after

    def test_custom_timestamp_preserved(self, pipeline_no_model):
        custom_ts = 1_700_000_000.0
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0, timestamp=custom_ts)
        assert reading["metadata"]["timestamp"] == custom_ts


# ---------------------------------------------------------------------------
# Tests: CSV Stream simulation
# ---------------------------------------------------------------------------
class TestSimulatedStream:

    def test_reads_first_row(self, pipeline_no_model):
        reading = pipeline_no_model.read_simulated_stream(0)
        assert reading != {}
        assert "channels" in reading

    def test_reads_middle_row(self, pipeline_no_model):
        reading = pipeline_no_model.read_simulated_stream(3)
        assert "channels" in reading
        # Row 3 should have higher temp than row 0 in our fixture
        temp3 = reading["channels"]["temperature"]["value"]
        reading0 = pipeline_no_model.read_simulated_stream(0)
        temp0 = reading0["channels"]["temperature"]["value"]
        assert temp3 >= temp0

    def test_index_wraps_around(self, pipeline_no_model):
        # Index beyond CSV length should wrap
        r_normal = pipeline_no_model.read_simulated_stream(0)
        r_wrapped = pipeline_no_model.read_simulated_stream(6)  # 6 rows, wraps to 0
        assert "channels" in r_wrapped

    def test_missing_csv_returns_empty(self, tmp_path):
        from layer4_iot.sensor_pipeline import IoTSensorPipeline
        pipeline = IoTSensorPipeline(models_dir=str(tmp_path), csv_path="/nonexistent/path.csv")
        result = pipeline.read_simulated_stream(0)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Telemetry analysis
# ---------------------------------------------------------------------------
class TestTelemetryAnalysis:

    def test_empty_payload_returns_stale(self, pipeline_no_model):
        result = pipeline_no_model.analyze_telemetry({})
        assert result["stale"] is True

    def test_normal_reading_low_confidence(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert result["iot_confidence"] < 0.5
        assert result["anomaly_flag"] is False

    def test_fire_reading_high_confidence(self, pipeline_no_model):
        # Extreme values — high temp, low humidity, high smoke
        reading = pipeline_no_model.format_sensor_reading(50.0, 10.0, 900.0)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert result["iot_confidence"] > 0.5
        assert result["anomaly_flag"] is True

    def test_stale_data_detected(self, pipeline_no_model):
        # Use a very old timestamp (10 minutes ago)
        old_ts = time.time() - 700.0
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0, timestamp=old_ts)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert result["stale"] is True

    def test_fresh_data_not_stale(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert result["stale"] is False

    def test_result_schema_complete(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(25.0, 60.0, 150.0)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert "iot_confidence" in result
        assert "latest_reading" in result
        assert "anomaly_flag" in result
        assert "stale" in result

    def test_confidence_in_valid_range(self, pipeline_no_model):
        reading = pipeline_no_model.format_sensor_reading(35.0, 40.0, 300.0)
        result = pipeline_no_model.analyze_telemetry(reading)
        assert 0.0 <= result["iot_confidence"] <= 1.0

    def test_isolation_forest_detects_anomaly(self, pipeline_with_isolation_forest):
        # Strongly anomalous reading
        reading = pipeline_with_isolation_forest.format_sensor_reading(60.0, 5.0, 950.0)
        result = pipeline_with_isolation_forest.analyze_telemetry(reading)
        # IsolationForest should flag this as anomaly
        assert result["anomaly_flag"] is True or result["iot_confidence"] > 0.3
