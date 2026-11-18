"""
tests/test_fusion.py
Unit and integration tests for the Fusion Engine
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.fusion_engine import fuse_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_weather(risk_level="Low", risk_score=0.1):
    return {"risk_level": risk_level, "risk_score": risk_score}

def make_thermal(detected=False, count=0, max_conf=0.0, points=None):
    return {
        "hotspot_detected": detected,
        "hotspot_count": count,
        "max_confidence": max_conf,
        "points": points or []
    }

def make_image(fire=False, confidence=0.1):
    return {"fire_detected": fire, "confidence": confidence}

def make_iot(conf=0.1, anomaly=False, stale=False):
    return {
        "iot_confidence": conf,
        "latest_reading": {"temp": 25.0, "humidity": 60.0, "smoke": 150.0},
        "anomaly_flag": anomaly,
        "stale": stale
    }


# ---------------------------------------------------------------------------
# Tests: Schema and basic behaviour
# ---------------------------------------------------------------------------
class TestFusionEngineSchema:

    def test_output_has_required_keys(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert "confidence_score" in result
        assert "alert_level" in result
        assert "reasoning" in result

    def test_confidence_score_in_0_to_100_range(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert 0.0 <= result["confidence_score"] <= 100.0

    def test_alert_level_is_valid_tier(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert result["alert_level"] in ("Watch", "Warning", "Confirmed Fire")

    def test_reasoning_contains_active_layers(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert "active_layers" in result["reasoning"]
        assert len(result["reasoning"]["active_layers"]) > 0

    def test_reasoning_contains_contributions(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert "contributions" in result["reasoning"]


# ---------------------------------------------------------------------------
# Tests: Alert tier scenarios
# ---------------------------------------------------------------------------
class TestAlertTiers:

    def test_all_low_scores_produces_watch(self):
        result = fuse_scores(
            make_weather("Low", 0.05),
            make_thermal(False, 0, 0.0),
            make_image(False, 0.05),
            make_iot(0.05, False)
        )
        assert result["alert_level"] == "Watch"
        assert result["confidence_score"] < 41.0

    def test_moderate_scores_produces_warning(self):
        result = fuse_scores(
            make_weather("Medium", 0.55),
            make_thermal(True, 1, 0.55),
            make_image(True, 0.60),
            make_iot(0.55, True)
        )
        assert result["alert_level"] in ("Warning", "Confirmed Fire")
        assert result["confidence_score"] >= 41.0

    def test_all_high_scores_produces_confirmed_fire(self):
        result = fuse_scores(
            make_weather("High", 0.95),
            make_thermal(True, 5, 0.95),
            make_image(True, 0.97),
            make_iot(0.92, True)
        )
        assert result["alert_level"] == "Confirmed Fire"
        assert result["confidence_score"] >= 71.0

    def test_single_high_signal_image_only(self):
        """Only vision layer active and confident; should at least be Watch."""
        result = fuse_scores(
            None,
            {"hotspot_detected": None, "error": "API down"},
            make_image(True, 0.95),
            None
        )
        assert result["alert_level"] in ("Watch", "Warning", "Confirmed Fire")
        assert result["confidence_score"] >= 0.0

    def test_boundary_score_41_is_warning(self):
        """Manually verify threshold: score at exactly 41 -> Warning."""
        # Tune inputs to aim for score near 41
        result = fuse_scores(
            make_weather("Medium", 0.41),
            make_thermal(True, 1, 0.41),
            make_image(True, 0.41),
            make_iot(0.41, True)
        )
        assert result["alert_level"] in ("Warning", "Confirmed Fire")

    def test_boundary_score_71_is_confirmed(self):
        """Score at exactly 71 -> Confirmed Fire."""
        result = fuse_scores(
            make_weather("High", 0.71),
            make_thermal(True, 3, 0.71),
            make_image(True, 0.71),
            make_iot(0.71, True)
        )
        assert result["alert_level"] == "Confirmed Fire"


# ---------------------------------------------------------------------------
# Tests: Missing / errored layer handling
# ---------------------------------------------------------------------------
class TestMissingLayerHandling:

    def test_none_weather_still_fuses(self):
        result = fuse_scores(None, make_thermal(True, 2, 0.8), make_image(True, 0.9), make_iot(0.85))
        assert result["alert_level"] in ("Warning", "Confirmed Fire")
        assert "weather" not in result["reasoning"]["active_layers"]

    def test_errored_thermal_still_fuses(self):
        thermal_error = {"hotspot_detected": None, "error": "API timeout", "hotspot_count": 0}
        result = fuse_scores(make_weather("High", 0.9), thermal_error, make_image(True, 0.9), make_iot(0.9))
        assert result["confidence_score"] > 0.0
        assert "thermal" not in result["reasoning"]["active_layers"]

    def test_stale_iot_excluded_from_fusion(self):
        stale_iot = make_iot(conf=0.95, anomaly=True, stale=True)
        result = fuse_scores(make_weather("Low", 0.1), make_thermal(), make_image(False, 0.1), stale_iot)
        assert "iot" not in result["reasoning"]["active_layers"]

    def test_all_layers_offline_returns_watch_with_error(self):
        result = fuse_scores(None, None, None, None)
        assert result["alert_level"] == "Watch"
        assert "error" in result["reasoning"]

    def test_weight_redistribution_flag_set(self):
        result = fuse_scores(None, make_thermal(True, 1, 0.8), make_image(True, 0.8), make_iot(0.8))
        assert result["reasoning"]["weight_redistributed"] is True

    def test_all_layers_active_no_redistribution(self):
        result = fuse_scores(make_weather(), make_thermal(), make_image(), make_iot())
        assert result["reasoning"]["weight_redistributed"] is False

    def test_two_layers_offline_score_not_zero(self):
        result = fuse_scores(None, None, make_image(True, 0.9), make_iot(0.85))
        assert result["confidence_score"] > 0.0

    def test_offline_layers_listed_in_reasoning(self):
        result = fuse_scores(None, make_thermal(), make_image(), make_iot())
        assert "weather" in result["reasoning"]["offline_layers"]


# ---------------------------------------------------------------------------
# Integration Test: Full Pipeline Simulation
# ---------------------------------------------------------------------------
class TestEndToEndFusion:

    def test_full_pipeline_all_clear(self):
        """Simulate all layers reporting no fire."""
        result = fuse_scores(
            make_weather("Low", 0.05),
            make_thermal(False, 0, 0.0),
            make_image(False, 0.08),
            make_iot(0.06, False)
        )
        assert result["alert_level"] == "Watch"
        assert result["confidence_score"] < 41.0

    def test_full_pipeline_partial_confirmation(self):
        """Vision and IoT agree there is a fire; thermal and weather are neutral."""
        result = fuse_scores(
            make_weather("Low", 0.15),
            make_thermal(False, 0, 0.0),
            make_image(True, 0.88),
            make_iot(0.82, True)
        )
        # Combined weight of vision+iot = 0.60, at high scores should be Warning or Confirmed
        assert result["alert_level"] in ("Warning", "Confirmed Fire")

    def test_full_pipeline_confirmed_fire(self):
        """All four sources agree: fire confirmed."""
        result = fuse_scores(
            make_weather("High", 0.92),
            make_thermal(True, 7, 0.95),
            make_image(True, 0.97),
            make_iot(0.91, True)
        )
        assert result["alert_level"] == "Confirmed Fire"
        assert result["confidence_score"] >= 71.0
        # Verify per-layer contributions all positive
        for layer_name, contrib in result["reasoning"]["contributions"].items():
            assert contrib["contribution_points"] > 0.0

    def test_full_pipeline_with_firms_down(self):
        """FIRMS API is down; remaining layers should still produce valid result."""
        thermal_down = {"hotspot_detected": None, "error": "API unreachable"}
        result = fuse_scores(
            make_weather("High", 0.88),
            thermal_down,
            make_image(True, 0.92),
            make_iot(0.87, True)
        )
        assert result["alert_level"] in ("Warning", "Confirmed Fire")
        assert result["confidence_score"] > 0.0
        assert result["reasoning"]["weight_redistributed"] is True
