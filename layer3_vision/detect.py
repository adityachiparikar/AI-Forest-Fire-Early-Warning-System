import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import logging
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from config import MODELS_DIR

logger = logging.getLogger(__name__)


class VisionModelNotFoundError(FileNotFoundError):
    """Exception raised when the vision model file is missing."""
    pass


class VisionDetector(ABC):
    """Abstract interface for vision models (CNN, YOLOv8, etc.) to allow future swap-outs."""

    @abstractmethod
    def detect_fire(self, image_path_or_file) -> dict:
        """
        Processes image and returns prediction.
        Expected Output: {"fire_detected": bool, "confidence": float}
        """
        pass


class KerasVisionDetector(VisionDetector):
    """Standard CNN classification implementation using TensorFlow/Keras."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = None
        self.class_names_path = os.path.join(models_dir, "class_names.json")
        self.class_names = {}
        self.model = None
        self.input_size = (300, 300)
        self.model_handles_rescaling = False
        self._load_model()

    def _load_model(self):
        """Loads keras CNN model safely."""
        candidate_paths = [
            os.path.join(self.models_dir, "wildfire_efficientnetb3_final.keras"),
            os.path.join(self.models_dir, "fire_cnn_model.h5"),
        ]
        self.model_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if self.model_path is None:
            raise VisionModelNotFoundError(
                f"Keras vision model artifact not found in {self.models_dir}. "
                "Please run training/train_vision.py or tools/generate_dummy_models.py first."
            )

        try:
            import tensorflow as tf
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            self.model = tf.keras.models.load_model(self.model_path)
            self.input_size = self._resolve_input_size()
            self.model_handles_rescaling = self._model_handles_input_rescaling()
            self.class_names = self._load_class_names()
            logger.info("Keras vision model loaded successfully.")
        except ImportError as ie:
            logger.error(f"ImportError details: {ie}", exc_info=True)
            logger.error("TensorFlow is not installed. Unable to load Keras CNN model.")
            raise ImportError(
                "TensorFlow is required to load and run the Keras vision model. "
                "Ensure tensorflow is installed, or use a dummy fallback model."
            ) from ie
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise

    def _resolve_input_size(self) -> tuple:
        """Reads the expected image size from the loaded Keras model."""
        input_shape = getattr(self.model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if isinstance(input_shape, tuple) and len(input_shape) >= 4:
            height, width = input_shape[1], input_shape[2]
            if isinstance(height, int) and isinstance(width, int):
                return (width, height)

        logger.warning("Unable to resolve Keras input shape. Falling back to 300x300.")
        return (300, 300)

    def _model_handles_input_rescaling(self) -> bool:
        """Returns True when preprocessing is already embedded in the Keras model."""
        for layer in getattr(self.model, "layers", []):
            if layer.__class__.__name__ == "Rescaling":
                return True
        return False

    def _load_class_names(self) -> dict:
        """Loads optional model class-name metadata keyed by output index."""
        if not os.path.exists(self.class_names_path):
            return {}

        try:
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                raw_names = json.load(f)
            return {int(index): str(name).lower() for index, name in raw_names.items()}
        except Exception as e:
            logger.warning(f"Unable to read vision class names: {e}")
            return {}

    def _interpret_prediction(self, prediction) -> dict:
        """Converts binary or multi-class model output into the app's fire schema."""
        scores = np.asarray(prediction[0], dtype=np.float32).reshape(-1)
        if scores.size == 0:
            raise ValueError("Model returned an empty prediction")

        if scores.size == 1:
            confidence = float(scores[0])
            return {
                "fire_detected": confidence > 0.5,
                "confidence": confidence,
                "predicted_label": "fire" if confidence > 0.5 else "no_fire",
            }

        predicted_index = int(np.argmax(scores))
        predicted_label = self.class_names.get(predicted_index, str(predicted_index))
        fire_indices = [idx for idx, name in self.class_names.items() if name == "fire"]
        fire_index = fire_indices[0] if fire_indices else 0
        fire_confidence = float(scores[fire_index])

        return {
            "fire_detected": predicted_label == "fire" or fire_confidence > 0.5,
            "confidence": fire_confidence,
            "predicted_label": predicted_label,
            "class_scores": {
                self.class_names.get(idx, str(idx)): float(score)
                for idx, score in enumerate(scores)
            },
        }

    def detect_fire(self, image_path_or_file) -> dict:
        """
        Preprocesses and predicts fire probability using the Keras model.
        Supports file paths or file-like objects (e.g., BytesIO from Streamlit upload).
        """
        if self.model is None:
            return {"fire_detected": False, "confidence": 0.0, "error": "Model not loaded"}

        try:
            if hasattr(image_path_or_file, "seek"):
                image_path_or_file.seek(0)

            img = Image.open(image_path_or_file).convert("RGB")
            img_resized = img.resize(self.input_size)
            img_array = np.array(img_resized, dtype=np.float32)
            if not self.model_handles_rescaling:
                img_array = img_array / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_batch, verbose=0)
            return self._interpret_prediction(prediction)

        except Exception as e:
            logger.error(f"Vision inference error: {e}")
            return {
                "fire_detected": False,
                "confidence": 0.0,
                "error": f"Failed to process image: {str(e)}"
            }


class DummyVisionDetector(VisionDetector):
    """Fallback vision detector used when TensorFlow/Keras cannot be imported/loaded."""

    def __init__(self, message="TensorFlow/Keras model missing"):
        self.message = message
        logger.warning(f"Using DummyVisionDetector: {message}")

    def detect_fire(self, image_path_or_file) -> dict:
        """Simulates prediction based on basic image properties or simple color checks."""
        try:
            if hasattr(image_path_or_file, "seek"):
                image_path_or_file.seek(0)

            img = Image.open(image_path_or_file).convert("RGB")
            img_resized = img.resize((32, 32))
            arr = np.array(img_resized)

            avg_r = np.mean(arr[:, :, 0])
            avg_g = np.mean(arr[:, :, 1])
            avg_b = np.mean(arr[:, :, 2])

            if avg_r > 130 and avg_g > 80 and avg_b < 80:
                confidence = 0.75
            else:
                confidence = 0.15

            return {
                "fire_detected": confidence > 0.5,
                "confidence": confidence,
                "warning": f"Fallback predictor: {self.message}"
            }
        except Exception as e:
            return {
                "fire_detected": False,
                "confidence": 0.0,
                "error": f"Fallback image parsing failed: {str(e)}"
            }