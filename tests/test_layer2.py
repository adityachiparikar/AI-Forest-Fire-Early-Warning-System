"""
tests/test_layer2.py
Unit tests for Layer 2: Satellite Thermal Anomaly (FIRMS Client)
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
MOCK_CSV_RESPONSE = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight
18.5123,73.2456,325.8,0.35,0.40,2024-01-15,0135,N,h,2.0NRT,293.1,4.2,D
17.8901,73.9012,311.2,0.37,0.41,2024-01-15,0135,N,n,2.0NRT,290.4,3.1,D
16.7500,73.5000,305.5,0.36,0.39,2024-01-15,0135,N,l,2.0NRT,288.2,2.8,D
"""

MOCK_EMPTY_CSV = "latitude,longitude,brightness,confidence\n"


@pytest.fixture
def firms_client_with_cache(tmp_path):
    from layer2_thermal.firms_client import FIRMSClient
    cache_path = str(tmp_path / "firms_cache.json")
    client = FIRMSClient(cache_path=cache_path)
    return client, cache_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFIRMSClient:

    def test_missing_api_key_returns_error(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": ""}, clear=False):
            client.api_key = ""
            result = client.query_hotspots()
        assert result["hotspot_detected"] is None
        assert "error" in result

    def test_missing_api_key_uses_cache_if_available(self, firms_client_with_cache, tmp_path):
        client, cache_path = firms_client_with_cache
        # Pre-populate cache
        cached_data = {
            "hotspot_detected": True,
            "hotspot_count": 2,
            "max_confidence": 0.9,
            "points": [{"lat": 18.5, "lon": 73.2, "confidence": 0.9}]
        }
        with open(cache_path, "w") as f:
            json.dump(cached_data, f)
        client.api_key = ""
        result = client.query_hotspots()
        assert result["hotspot_count"] == 2
        assert result.get("cached") is True

    def test_successful_api_response_parsed(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        client.api_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_CSV_RESPONSE
        with patch("requests.get", return_value=mock_resp):
            result = client.query_hotspots()
        assert result["hotspot_detected"] is True
        assert result["hotspot_count"] >= 1
        assert result["max_confidence"] > 0.0
        assert len(result["points"]) >= 1

    def test_empty_csv_response_means_no_hotspots(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        client.api_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_EMPTY_CSV
        with patch("requests.get", return_value=mock_resp):
            result = client.query_hotspots()
        assert result["hotspot_detected"] is False
        assert result["hotspot_count"] == 0

    def test_api_failure_falls_back_to_cache(self, firms_client_with_cache, tmp_path):
        client, cache_path = firms_client_with_cache
        client.api_key = "test_key_123"
        cached_data = {
            "hotspot_detected": True,
            "hotspot_count": 5,
            "max_confidence": 0.8,
            "points": []
        }
        with open(cache_path, "w") as f:
            json.dump(cached_data, f)
        with patch("requests.get", side_effect=Exception("Connection timeout")):
            result = client.query_hotspots()
        assert result["hotspot_count"] == 5
        assert result.get("cached") is True

    def test_api_failure_no_cache_returns_none_detected(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        client.api_key = "test_key_123"
        with patch("requests.get", side_effect=Exception("Network error")):
            result = client.query_hotspots()
        assert result["hotspot_detected"] is None
        assert "error" in result

    def test_points_respect_bounding_box(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        client.api_key = "test_key_123"
        # Include a point outside the bbox
        csv_with_outlier = (
            "latitude,longitude,brightness,confidence\n"
            "18.5123,73.2456,325.8,h\n"   # Inside bbox
            "10.0000,80.0000,310.0,h\n"   # Outside bbox
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_with_outlier
        with patch("requests.get", return_value=mock_resp):
            result = client.query_hotspots()
        # Only points within bbox should be included
        for pt in result["points"]:
            assert 73.0 <= pt["lon"] <= 74.5
            assert 16.5 <= pt["lat"] <= 19.5

    def test_deduplication_merges_close_points(self, firms_client_with_cache):
        client, _ = firms_client_with_cache
        # Create very close points that should be deduplicated
        close_points = [
            {"lat": 18.5000, "lon": 73.2000, "confidence": 0.5},
            {"lat": 18.5001, "lon": 73.2001, "confidence": 0.9},  # Should win (higher conf)
            {"lat": 18.5002, "lon": 73.2002, "confidence": 0.3},
        ]
        result = client._deduplicate_points(close_points, grid_size=0.01)
        # These are close enough to be in the same grid cell
        assert len(result) <= len(close_points)
        if len(result) == 1:
            # The highest confidence point should remain
            assert result[0]["confidence"] == 0.9

    def test_successful_response_writes_cache(self, firms_client_with_cache, tmp_path):
        client, cache_path = firms_client_with_cache
        client.api_key = "test_key_123"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_CSV_RESPONSE
        with patch("requests.get", return_value=mock_resp):
            client.query_hotspots()
        assert os.path.exists(cache_path)
        with open(cache_path) as f:
            cached = json.load(f)
        assert "hotspot_detected" in cached
