"""
tests/test_layer3.py
Unit tests for Layer 3: Image / Video Fire Detection
"""
import os
import sys
import io
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_test_image(color=(200, 100, 50), size=(224, 224)) -> io.BytesIO:
    """Creates a minimal in-memory JPEG image for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def create_fire_colored_image() -> io.BytesIO:
    """Creates an image with strong red/orange tones (fire-like)."""
    return create_test_image(color=(220, 90, 20))


def create_forest_colored_image() -> io.BytesIO:
    """Creates an image with green tones (no fire)."""
    return create_test_image(color=(40, 120, 40))


# ---------------------------------------------------------------------------
# Tests: DummyVisionDetector (no TF required)
# ---------------------------------------------------------------------------
class TestDummyVisionDetector:

    def test_returns_correct_schema_on_valid_image(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        result = detector.detect_fire(create_test_image())
        assert "fire_detected" in result
        assert "confidence" in result
        assert isinstance(result["fire_detected"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_fire_colored_image_has_higher_confidence(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        fire_result = detector.detect_fire(create_fire_colored_image())
        forest_result = detector.detect_fire(create_forest_colored_image())
        # Fire-colored image should produce higher confidence than green forest
        assert fire_result["confidence"] >= forest_result["confidence"]

    def test_invalid_image_path_returns_error(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        result = detector.detect_fire("/nonexistent/path/to/image.jpg")
        assert "error" in result
        assert result["fire_detected"] is False

    def test_corrupted_bytes_returns_error(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        buf = io.BytesIO(b"this is not an image")
        result = detector.detect_fire(buf)
        assert "error" in result

    def test_confidence_is_float_between_0_and_1(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        result = detector.detect_fire(create_test_image(color=(128, 128, 128)))
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_different_image_sizes_handled(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        small = create_test_image(size=(32, 32))
        large = create_test_image(size=(1920, 1080))
        r1 = detector.detect_fire(small)
        r2 = detector.detect_fire(large)
        assert "fire_detected" in r1
        assert "fire_detected" in r2

    def test_png_format_works(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        img = Image.new("RGB", (224, 224), color=(200, 80, 20))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        result = detector.detect_fire(buf)
        assert "fire_detected" in result

    def test_rgba_image_converted_to_rgb(self):
        from layer3_vision.detect import DummyVisionDetector
        detector = DummyVisionDetector()
        img = Image.new("RGBA", (224, 224), color=(200, 80, 20, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        result = detector.detect_fire(buf)
        assert "fire_detected" in result


# ---------------------------------------------------------------------------
# Tests: KerasVisionDetector (mocked model)
# ---------------------------------------------------------------------------
class TestKerasVisionDetectorMocked:

    def test_raises_on_missing_model(self, tmp_path):
        from layer3_vision.detect import KerasVisionDetector, VisionModelNotFoundError
        with pytest.raises(VisionModelNotFoundError):
            KerasVisionDetector(models_dir=str(tmp_path))

    def test_predict_calls_model_predict(self, tmp_path):
        """Uses a mock to simulate a loaded Keras model."""
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            pytest.skip("TensorFlow not installed; skipping Keras model tests.")

        from layer3_vision.detect import KerasVisionDetector
        import unittest.mock as mock

        # Create fake h5 model file so path check passes
        model_file = tmp_path / "fire_cnn_model.h5"
        model_file.write_bytes(b"fake")

        mock_keras_model = mock.MagicMock()
        mock_keras_model.predict.return_value = np.array([[0.85]])

        with mock.patch("tensorflow.keras.models.load_model", return_value=mock_keras_model):
            detector = KerasVisionDetector(models_dir=str(tmp_path))
            result = detector.detect_fire(create_test_image())
            assert "fire_detected" in result
            assert "confidence" in result
            assert result["confidence"] == pytest.approx(0.85, abs=0.01)
