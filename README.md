# Multi-Modal Forest Fire Early-Warning System
## Western Maharashtra / Sahyadri Belt

A multi-modal, AI-powered early-warning system designed to detect forest fires
across the Western Ghats (Sahyadri Belt) of Maharashtra, India. The system fuses
evidence from four independent sensor modalities -- weather data, satellite
thermal imagery, visual CNN classification, and IoT ground sensors -- into a
single confidence score with tiered alerts. Built with ethical AI principles
following IEEE 7000-series standards.

**Target Region:** Western Maharashtra (Sahyadri Belt)
- Bounding Box: 73.0E to 74.5E, 16.5N to 19.5N
- Center: 18.0N, 73.7E


---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Requirements](#system-requirements)
3. [Setup Instructions](#setup-instructions)
4. [Project Structure](#project-structure)
5. [Layer Descriptions](#layer-descriptions)
6. [Fusion Engine](#fusion-engine)
7. [Training Models](#training-models)
8. [IEEE Standards Compliance](#ieee-standards-compliance)
9. [Testing](#testing)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)


---

## Architecture Overview

The system is organized into four independent sensing layers that feed into a
central Fusion Engine, which produces a unified risk assessment displayed on a
Streamlit dashboard.

```
+---------------------------------------------------------------------+
|                          INPUT SOURCES                               |
|                                                                      |
|  [Weather Data]  [NASA FIRMS]  [Camera Images]  [IoT Sensors]       |
|   temp, RH,       VIIRS/MODIS    RGB frames       temp, humidity,   |
|   wind, rain,      satellite     from field        smoke (PPM)      |
|   FFMC, DMC...     hotspots      cameras           via MQTT/CSV     |
+--------+-------------+--------------+----------------+--------------+
         |             |              |                |
         v             v              v                v
+--------+---+ +------+------+ +-----+-------+ +------+------+
|  LAYER 1   | |  LAYER 2    | |  LAYER 3    | |  LAYER 4    |
|  Weather   | |  Satellite  | |  Vision     | |  IoT Ground |
|  Risk      | |  Thermal    | |  CNN        | |  Sensors    |
|            | |             | |             | |             |
| RandomFor- | | NASA FIRMS  | | Keras CNN   | | Isolation   |
| est Classi-| | VIIRS/MODIS | | 224x224 RGB | | Forest +    |
| fier (pkl) | | API Client  | | fire detect | | Thresholds  |
|            | |             | |             | |             |
| Output:    | | Output:     | | Output:     | | Output:     |
| risk_score | | max_confid- | | confidence  | | iot_confid- |
| risk_level | | ence,       | | fire_detec- | | ence,       |
|  (0.0-1.0) | | hotspot_cnt | | ted (bool)  | | anomaly_    |
+-----+------+ +------+-----+ +------+------+ | flag        |
      |               |              |         +------+------+
      |               |              |                |
      +-------+-------+------+-------+-------+-------+
              |              |               |
              v              v               v
      +-------+--------------+---------------+--------+
      |                FUSION ENGINE                   |
      |                                                |
      |  Weighted combination of all available layers  |
      |  with proportional weight redistribution for   |
      |  any offline/missing modalities.               |
      |                                                |
      |  Weights: Weather=0.15, Thermal=0.25,          |
      |           Image=0.30, IoT=0.30                 |
      |                                                |
      |  Output: confidence_score (0-100%)             |
      |          alert_level (Watch/Warning/Confirmed) |
      +-------------------------+----------------------+
                                |
                                v
                  +-------------+-------------+
                  |     STREAMLIT DASHBOARD    |
                  |                            |
                  |  - Interactive Folium map   |
                  |  - Real-time risk gauge     |
                  |  - Layer-by-layer breakdown |
                  |  - Satellite hotspot overlay|
                  |  - Alert history log        |
                  +----------------------------+
```


---

## System Requirements

| Requirement     | Minimum Version | Notes                                     |
|-----------------|-----------------|-------------------------------------------|
| Python          | 3.9+            | Tested on 3.9, 3.10, 3.11                |
| pip             | 21.0+           | For installing dependencies               |
| TensorFlow      | 2.11+           | Optional; fallback detector used if absent|
| Operating System| Windows / Linux / macOS | All platforms supported          |
| RAM             | 4 GB+           | 8 GB recommended for CNN inference        |
| Disk Space      | ~500 MB         | For models and cached data                |

**Note:** TensorFlow is optional. If not installed, the Vision CNN layer will
use a `DummyVisionDetector` fallback that performs basic color-heuristic fire
detection. Full accuracy requires TensorFlow with the trained Keras model.


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd forest-fire-prediction
```

### 2. Create a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package            | Purpose                                    |
|--------------------|--------------------------------------------|
| streamlit          | Dashboard web UI                           |
| pandas             | Data manipulation and CSV handling          |
| numpy              | Numerical operations                       |
| scikit-learn       | Weather model and IoT anomaly detection    |
| tensorflow         | Vision CNN model (optional, platform-aware) |
| paho-mqtt          | MQTT client for real IoT sensor ingestion  |
| requests           | NASA FIRMS API HTTP calls                  |
| folium             | Interactive map rendering                  |
| streamlit-folium   | Folium integration within Streamlit        |
| python-dotenv      | Load environment variables from .env       |
| pytest             | Test runner                                |

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file and add your NASA FIRMS Map Key:

```
FIRMS_MAP_KEY=your_actual_api_key_here
```

**Get a free API key:** Visit https://firms.modaps.eosdis.nasa.gov/api/map_key/
and register for a free NASA FIRMS Map Key. The key is used to query satellite
thermal hotspot data (VIIRS S-NPP, 375m resolution) for the configured region.

If you do not have a key, the system will still function but the Satellite
Thermal layer will operate in offline/cached mode or return no data.

### 5. Generate Dummy Models

Before first run, generate placeholder ML models for development and testing:

```bash
python tools/generate_dummy_models.py
```

This creates the following files in the `models/` directory:

| File                      | Description                              |
|---------------------------|------------------------------------------|
| `weather_risk_model.pkl`  | RandomForest fire risk classifier        |
| `le_month.pkl`            | LabelEncoder for month feature           |
| `le_day.pkl`              | LabelEncoder for day feature             |
| `fire_cnn_model.h5`       | Tiny CNN model (requires TensorFlow)     |
| `iot_anomaly_model.pkl`   | IsolationForest anomaly detector         |

**Note:** If TensorFlow is not installed, the CNN model (`fire_cnn_model.h5`)
will be skipped with a warning. The application will use the
`DummyVisionDetector` fallback at runtime.

### 6. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.


---

## Project Structure

```
forest-fire-prediction/
|
|-- .env.example              # Template for environment variables
|-- config.py                 # Central configuration (region, weights, paths)
|-- requirements.txt          # Python package dependencies
|-- app.py                    # Streamlit dashboard entry point
|-- README.md                 # This file
|
|-- layer1_weather/           # LAYER 1: Weather Risk Analysis
|   |-- __init__.py
|   |-- predict.py            # WeatherRiskPredictor (RandomForest)
|
|-- layer2_thermal/           # LAYER 2: Satellite Thermal Detection
|   |-- __init__.py
|   |-- firms_client.py       # FIRMSClient (NASA FIRMS API + cache)
|
|-- layer3_vision/            # LAYER 3: Vision / CNN Fire Detection
|   |-- __init__.py
|   |-- detect.py             # KerasVisionDetector + DummyVisionDetector
|
|-- layer4_iot/               # LAYER 4: IoT Ground Sensor Pipeline
|   |-- __init__.py
|   |-- sensor_pipeline.py    # IoTSensorPipeline (MQTT + IsolationForest)
|
|-- fusion/                   # FUSION ENGINE: Multi-modal score aggregation
|   |-- __init__.py
|   |-- fusion_engine.py      # fuse_scores() with weight redistribution
|
|-- training/                 # Model training scripts
|   |-- __init__.py
|   |-- train_weather.py      # Train weather risk model on UCI dataset
|
|-- tools/                    # Utility scripts
|   |-- generate_dummy_models.py  # Generate placeholder models for dev/test
|
|-- models/                   # Trained model artifacts (generated)
|   |-- weather_risk_model.pkl
|   |-- le_month.pkl
|   |-- le_day.pkl
|   |-- iot_anomaly_model.pkl
|   |-- fire_cnn_model.h5     # Only if TensorFlow is installed
|
|-- data/                     # Data directory (generated/cached)
|   |-- firms_cache.json      # Cached NASA FIRMS API responses
|   |-- iot_sim/
|       |-- sensor_stream.csv # Simulated IoT sensor telemetry
|
|-- tests/                    # Test suite
|   |-- __init__.py
```


---

## Layer Descriptions

### Layer 1: Weather Risk Analysis

**Module:** `layer1_weather/predict.py`
**Model:** RandomForest Classifier (`weather_risk_model.pkl`)

Analyzes meteorological conditions to assess fire danger. Uses the Canadian
Forest Fire Weather Index (FWI) System components as input features:

| Feature | Description                                         |
|---------|-----------------------------------------------------|
| FFMC    | Fine Fuel Moisture Code (surface litter moisture)   |
| DMC     | Duff Moisture Code (moderate-depth organic layer)   |
| DC      | Drought Code (deep organic layer dryness)           |
| ISI     | Initial Spread Index (expected fire spread rate)    |
| temp    | Temperature in Celsius                              |
| RH      | Relative humidity (%)                               |
| wind    | Wind speed (km/h)                                   |
| rain    | Rainfall (mm)                                       |
| month   | Month of the year (categorical, label-encoded)      |
| day     | Day of the week (categorical, label-encoded)        |

**Output:** `risk_score` (0.0 to 1.0) and `risk_level` (Low / Medium / High)

- Low:    risk_score < 0.35
- Medium: 0.35 <= risk_score < 0.70
- High:   risk_score >= 0.70

The model includes input validation and safe defaults for missing fields.

---

### Layer 2: Satellite Thermal Detection

**Module:** `layer2_thermal/firms_client.py`
**Data Source:** NASA FIRMS API (VIIRS S-NPP Near Real-Time, 375m resolution)

Queries the NASA Fire Information for Resource Management System (FIRMS) for
active thermal hotspots within the configured Sahyadri bounding box. Features:

- **Dual-source querying:** Tries VIIRS_SNPP_NRT first, falls back to MODIS_NRT
- **Offline resilience:** Caches successful API responses to `firms_cache.json`
  and falls back to cached data when the API is unreachable
- **Confidence parsing:** Handles both categorical (H/N/L) and numeric
  confidence values from the FIRMS CSV response
- **Deduplication:** Applies grid-based deduplication (0.01-degree resolution)
  to prevent alert fatigue from closely-spaced hotspot clusters (IEEE 7010)
- **Bounding box filtering:** Validates all returned points against the
  configured regional bounding box

**Output:** `hotspot_detected` (bool), `hotspot_count` (int),
`max_confidence` (0.0-1.0), `points` (list of lat/lon/confidence)

---

### Layer 3: Vision CNN Fire Detection

**Module:** `layer3_vision/detect.py`
**Model:** Keras Sequential CNN (`fire_cnn_model.h5`)

Classifies uploaded images as fire or non-fire using a Convolutional Neural
Network. The module provides two implementations behind a common
`VisionDetector` abstract interface:

| Detector                | When Used                          | Method                |
|-------------------------|------------------------------------|-----------------------|
| `KerasVisionDetector`   | TensorFlow installed + model found | CNN inference          |
| `DummyVisionDetector`   | TensorFlow absent or model missing | Color heuristic       |

**Image preprocessing pipeline:**
1. Open image (supports file paths and file-like objects from Streamlit uploads)
2. Convert to RGB
3. Resize to 224 x 224 pixels
4. Normalize pixel values to [0, 1] range
5. Add batch dimension -> shape (1, 224, 224, 3)
6. Run sigmoid-activated final layer -> confidence 0.0 to 1.0

**Output:** `fire_detected` (bool, threshold 0.5), `confidence` (0.0-1.0)

---

### Layer 4: IoT Ground Sensors

**Module:** `layer4_iot/sensor_pipeline.py`
**Model:** IsolationForest anomaly detector (`iot_anomaly_model.pkl`)

Processes telemetry from ground-deployed sensor nodes (ESP32-based) using a
two-stage detection pipeline:

**Stage 1 -- Threshold Rules (Interpretable baseline):**

| Sensor         | Fire Indicator Threshold | Scoring Formula                      |
|----------------|--------------------------|--------------------------------------|
| Temperature    | > 40 C                   | (temp - 30) / 15, clamped [0, 1]    |
| Humidity       | < 25%                    | (45 - humidity) / 30, clamped [0, 1] |
| Smoke (PPM)    | > 250 PPM                | (smoke - 200) / 600, clamped [0, 1] |

Threshold confidence = (temp_score * 0.3) + (humidity_score * 0.2) + (smoke_score * 0.5)

**Stage 2 -- ML Anomaly Detection:**
- IsolationForest: Labels data points as normal (1) or anomaly (-1)
- Decision function score normalized to a probability-like confidence
- Supports Autoencoder (.h5) as an alternative model (reconstruction error)

**Data sources:**
- **Simulated:** Reads from `data/iot_sim/sensor_stream.csv`
- **Live:** MQTT subscriber on `sahyadri/iot/sensors` topic (IEEE 2413)

**Sensor payload format (IEEE 1451 TEDS):**
```json
{
  "metadata": {
    "sensor_platform": "ESP32-Sahyadri-Node",
    "standard": "IEEE-1451-TEDS",
    "timestamp": 1689000000.0
  },
  "channels": {
    "temperature": {"value": 35.2, "unit": "Celsius"},
    "humidity": {"value": 28.0, "unit": "Percent"},
    "smoke_concentration": {"value": 310.0, "unit": "PPM"}
  }
}
```

**Output:** `iot_confidence` (0.0-1.0), `anomaly_flag` (bool), `stale` (bool)

Stale data (timestamp older than 5 minutes) is automatically flagged and
excluded from fusion to prevent misleading readings.


---

## Fusion Engine

**Module:** `fusion/fusion_engine.py`
**Function:** `fuse_scores(weather_risk, thermal, image, iot) -> dict`

The Fusion Engine combines evidence from all four layers into a single
confidence score using weighted averaging with proportional redistribution
for any offline modalities.

### Layer Weights

| Layer            | Key      | Default Weight | Description                     |
|------------------|----------|----------------|---------------------------------|
| Weather Risk     | weather  | 0.15           | Meteorological conditions       |
| Satellite Thermal| thermal  | 0.25           | NASA FIRMS hotspot data         |
| Vision CNN       | image    | 0.30           | Camera image classification     |
| IoT Sensors      | iot      | 0.30           | Ground sensor anomaly detection |
| **Total**        |          | **1.00**       |                                 |

Weights are configured in `config.py` under `FUSION_WEIGHTS` and must sum to
1.0.

### Weight Redistribution

When one or more layers are offline (missing data, API errors, model not
loaded), the remaining layers' weights are scaled proportionally so they still
sum to 1.0:

```
adjusted_weight = original_weight / sum(active_layers_original_weights)
```

Example: If the Satellite layer (0.25) is offline, the remaining layers
(Weather=0.15, Image=0.30, IoT=0.30, total=0.75) are rescaled:
- Weather: 0.15 / 0.75 = 0.20
- Image:   0.30 / 0.75 = 0.40
- IoT:     0.30 / 0.75 = 0.40

### Alert Tiers

The final confidence score (0-100%) is classified into three alert levels
using asymmetric thresholds biased toward caution (IEEE 7000):

| Alert Level     | Score Range  | Response Action                            |
|-----------------|--------------|--------------------------------------------|
| Watch           | 0% - 40%    | Normal monitoring, no immediate action     |
| Warning         | 41% - 70%   | Elevated caution, deploy ground checks     |
| Confirmed Fire  | 71% - 100%  | Immediate automated alert propagation      |

**Design rationale (IEEE 7000 Ethical Compliance):** The cost of a false
negative (undetected fire) is catastrophically higher than a false positive
(temporary false alarm). Therefore, the Warning threshold is deliberately
lowered to 41% and the Confirmed threshold to 71% to trigger early
mobilization and minimize response latency.


---

## Training Models

### Quick Start (Dummy Models)

For development and testing, generate placeholder models:

```bash
python tools/generate_dummy_models.py
```

This creates small, randomly-initialized models that allow the full pipeline
to run end-to-end. These are NOT suitable for production fire detection.

### Full Training

Production-quality models should be trained using the scripts in the
`training/` directory:

**Weather Risk Model:**

```bash
python training/train_weather.py
```

- Trains a `RandomForestClassifier` (200 estimators) on the UCI Forest Fires
  dataset
- Expects the dataset at `training/data/forest_fires.csv`
- Outputs: `weather_risk_model.pkl`, `le_month.pkl`, `le_day.pkl` in `models/`
- Performs 80/20 stratified train/test split with evaluation metrics

**Vision CNN Model:**

Training scripts for the fire detection CNN can be added to
`training/train_vision.py`. The model architecture expects:
- Input: 224 x 224 x 3 RGB images
- Output: Single sigmoid neuron (binary classification)
- Recommended datasets: FLAME, FiSmo, or custom aerial/ground fire imagery

**IoT Anomaly Model:**

The IsolationForest model can be retrained on site-specific sensor telemetry
data. Collect baseline (non-fire) temperature, humidity, and smoke readings
from deployed ESP32 nodes to calibrate the anomaly detection thresholds.


---

## IEEE Standards Compliance

This system is designed with adherence to the following IEEE standards:

### IEEE 7000 -- Ethical Design of Autonomous and Intelligent Systems

- **Asymmetric risk thresholds:** Alert tiers are biased toward caution to
  minimize false negatives (undetected fires), acknowledging that the
  consequences of missing a real fire far outweigh the cost of a false alarm.
- **Transparency:** The fusion engine returns full reasoning, including
  per-layer contributions and weight redistribution details.
- **Safe defaults:** When all layers are offline, the system defaults to
  "Watch" status rather than signaling no risk.

### IEEE 7010 -- Well-being Impact Assessment

- **Alert fatigue mitigation:** Satellite hotspot data is deduplicated using
  grid-based spatial clustering (0.01-degree resolution) to prevent redundant
  notification storms from closely-spaced thermal anomalies.
- **Stale data handling:** IoT sensor readings older than 5 minutes are
  flagged as stale and excluded from fusion calculations to avoid misleading
  risk assessments.

### IEEE 2413 -- Standard for an Architectural Framework for the IoT

- **Three-layer IoT architecture:** The sensor pipeline implements the
  Sensing -> Network -> Application separation:
  - Sensing: Raw telemetry acquisition from ESP32 nodes
  - Network: MQTT-based edge gateway for real-time data ingestion
  - Application: Anomaly detection and fusion integration

### IEEE 1451 -- Smart Transducer Interface Standard

- **TEDS-compliant payloads:** All sensor readings are formatted into
  standardized, self-describing Transducer Electronic Data Sheets (TEDS) with
  metadata (platform, standard, timestamp) and typed channel values (unit,
  value).


---

## Testing

The project uses `pytest` for testing. Test files are located in the `tests/`
directory.

### Running All Tests

```bash
pytest
```

### Running Tests with Verbose Output

```bash
pytest -v
```

### Running a Specific Test File

```bash
pytest tests/test_fusion.py
```

### Running Tests with Coverage (requires pytest-cov)

```bash
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

### Writing Tests

Add new test files to the `tests/` directory following the naming convention
`test_<module>.py`. Example test structure:

```python
from fusion.fusion_engine import fuse_scores

def test_fuse_all_layers_active():
    weather = {"risk_score": 0.6, "risk_level": "Medium"}
    thermal = {"hotspot_detected": True, "max_confidence": 0.8, "hotspot_count": 3}
    image = {"fire_detected": True, "confidence": 0.75}
    iot = {"iot_confidence": 0.5, "anomaly_flag": True, "stale": False}

    result = fuse_scores(weather, thermal, image, iot)
    assert 0 <= result["confidence_score"] <= 100
    assert result["alert_level"] in ["Watch", "Warning", "Confirmed Fire"]

def test_fuse_all_layers_offline():
    result = fuse_scores({}, {}, {}, {})
    assert result["alert_level"] == "Watch"
    assert result["confidence_score"] == 0.0
```


---

## Configuration

All core configuration is centralized in `config.py`.

### Changing the Target Region

To monitor a different geographic area, update the `REGION` dictionary in
`config.py`:

```python
REGION = {
    "name": "Your Region Name",
    "bbox": {  # west, south, east, north (decimal degrees)
        "west": 73.0,
        "south": 16.5,
        "east": 74.5,
        "north": 19.5
    },
    "center": {"lat": 18.0, "lon": 73.7}
}
```

The bounding box is used by:
- The NASA FIRMS API client to query satellite hotspots
- The Folium map to set the default view
- The deduplication grid for hotspot clustering

### Adjusting Fusion Weights

Modify `FUSION_WEIGHTS` to change how much each layer contributes to the
final score. Weights must sum to 1.0:

```python
FUSION_WEIGHTS = {
    "weather": 0.15,
    "thermal": 0.25,
    "image":   0.30,
    "iot":     0.30
}
```

### Adjusting Alert Thresholds

Modify the alert threshold constants to change the sensitivity of the
tiered alert system:

```python
ALERT_THRESHOLD_WATCH     = 0.0    # 0% - 40%
ALERT_THRESHOLD_WARNING   = 41.0   # 41% - 70%
ALERT_THRESHOLD_CONFIRMED = 71.0   # 71% - 100%
```

Lower thresholds increase sensitivity (more alerts), higher thresholds
decrease sensitivity (fewer alerts).

### Directory Paths

Model and data directories are auto-created on import:

```python
MODELS_DIR     = os.path.join(BASE_DIR, "models")
DATA_DIR       = os.path.join(BASE_DIR, "data")
IOT_SIM_DIR    = os.path.join(DATA_DIR, "iot_sim")
```


---

## Troubleshooting

### Missing Model Files

**Symptom:** `WeatherModelNotFoundError` or `VisionModelNotFoundError` on
startup.

**Cause:** Model artifacts have not been generated or trained yet.

**Fix:**
```bash
python tools/generate_dummy_models.py
```
This generates all required model files in the `models/` directory. For
production, use the training scripts in `training/` instead.

---

### NASA FIRMS API Key Issues

**Symptom:** "NASA FIRMS API Key not configured" warning; satellite layer
returns no data.

**Cause:** The `FIRMS_MAP_KEY` environment variable is missing, empty, or
still set to the placeholder value `your_nasa_firms_map_key_here`.

**Fix:**
1. Get a free API key from https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Copy `.env.example` to `.env` if you have not already
3. Replace the placeholder value with your actual key:
   ```
   FIRMS_MAP_KEY=your_actual_key_here
   ```
4. Restart the application

**Note:** The system will still function without a valid FIRMS key. It will
use cached data if available, or the satellite thermal layer will simply
be excluded from the fusion calculation with its weight redistributed to
the remaining active layers.

---

### TensorFlow Not Installed

**Symptom:** "[WARN] TensorFlow not installed. Skipping Keras
fire_cnn_model.h5 creation." during model generation, or Vision layer uses
`DummyVisionDetector` at runtime.

**Cause:** TensorFlow is not installed in the current Python environment.

**Fix (if TensorFlow is needed):**
```bash
# Standard installation (Windows/Linux)
pip install tensorflow>=2.11.0

# Apple Silicon Mac
pip install tensorflow-macos>=2.11.0
```

**Note:** TensorFlow is optional. Without it:
- `generate_dummy_models.py` will skip creating `fire_cnn_model.h5`
- The Vision layer will automatically use the `DummyVisionDetector`, which
  applies a basic color-heuristic (red/orange channel dominance) instead of
  CNN inference
- All other layers and the Fusion Engine will work normally

---

### IoT Sensor Data Not Found

**Symptom:** "Simulated CSV stream not found" error in the IoT layer.

**Cause:** The simulated sensor data file `data/iot_sim/sensor_stream.csv`
does not exist.

**Fix:** Create the simulated data file with the expected columns:
```csv
temp,humidity,smoke
25.3,62.1,145.0
26.1,58.4,152.3
...
```

Or ensure the IoT simulation is running and generating data into the expected
path configured in `config.py` (`IOT_SIM_CSV`).

---

### Streamlit Won't Start

**Symptom:** `ModuleNotFoundError` or `streamlit: command not found`.

**Fix:**
1. Ensure you have activated your virtual environment
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Run with: `streamlit run app.py` or `python -m streamlit run app.py`

---

### Stale IoT Data Warning

**Symptom:** IoT layer reports `"stale": true` and is excluded from fusion.

**Cause:** The most recent sensor reading timestamp is older than 5 minutes
(300 seconds).

**Fix:** This is expected behavior for safety (IEEE 2413). Ensure the
simulated CSV has recent timestamps, or verify that the MQTT sensor feed
is actively transmitting data.


---

## License

*License information to be added.*

---

**Developed for early-warning fire detection across the Western Ghats
(Sahyadri) biodiversity hotspot of Maharashtra, India.**
