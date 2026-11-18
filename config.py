import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default region configuration for Western Maharashtra (Sahyadri Belt)
REGION = {
    "name": "Western Maharashtra (Sahyadri Belt)",
    "bbox": {  # west, south, east, north
        "west": 73.0,
        "south": 16.5,
        "east": 74.5,
        "north": 19.5
    },
    "center": {"lat": 18.0, "lon": 73.7}
}

# Preset regions: fire-prone forest zones across India (and a few global examples)
_r = lambda w, s, e, n, lat, lon: {
    "bbox": {"west": w, "south": s, "east": e, "north": n},
    "center": {"lat": lat, "lon": lon},
}
PRESET_REGIONS = {
    "Western Maharashtra (Sahyadri Belt)": _r(73.0, 16.5, 74.5, 19.5, 18.0, 73.7),
    "Melghat Tiger Reserve, Maharashtra":  _r(76.8, 21.3, 77.5, 21.8, 21.55, 77.15),
    "Tadoba-Andhari Tiger Reserve, Maharashtra": _r(79.2, 20.0, 79.6, 20.4, 20.2, 79.4),
    "Bandipur National Park, Karnataka":   _r(76.1, 11.5, 76.7, 12.0, 11.75, 76.4),
    "Jim Corbett National Park, Uttarakhand": _r(78.7, 29.3, 79.2, 29.7, 29.5, 78.95),
    "Simlipal National Park, Odisha":      _r(86.2, 21.5, 86.8, 22.1, 21.8, 86.5),
    "Nilgiri Biosphere Reserve, Tamil Nadu": _r(76.0, 11.0, 77.0, 11.8, 11.4, 76.5),
    "Amazon Rainforest, Brazil (Sample)":  _r(-62.0, -5.0, -58.0, -1.0, -3.0, -60.0),
    "California, USA (Sample)":            _r(-122.5, 36.5, -118.5, 40.0, 38.25, -120.5),
    "Custom Location": None,  # Sentinel: user will enter coordinates manually
}

# Fusion Engine Weights (must sum to 1.0)
FUSION_WEIGHTS = {
    "weather": 0.15,
    "thermal": 0.25,
    "image": 0.30,
    "iot": 0.30
}

# Thresholds for Alert levels (IEEE 7000 compliance: biased towards safety)
ALERT_THRESHOLD_WATCH = 0.0      # 0% - 40%
ALERT_THRESHOLD_WARNING = 41.0    # 41% - 70%
ALERT_THRESHOLD_CONFIRMED = 71.0  # 71% - 100%

# Folder Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIRMS_CACHE_PATH = os.path.join(DATA_DIR, "firms_cache.json")
IOT_SIM_DIR = os.path.join(DATA_DIR, "iot_sim")
IOT_SIM_CSV = os.path.join(IOT_SIM_DIR, "sensor_stream.csv")

# Create folders if not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IOT_SIM_DIR, exist_ok=True)

# NASA FIRMS API Endpoints
# Using the NASA FIRMS map key endpoint
FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{west},{south},{east},{north}/{range}/{date}"
