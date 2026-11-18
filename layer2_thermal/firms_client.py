import os
import json
import logging
import requests
import datetime
from dotenv import load_dotenv
from config import REGION, FIRMS_API_URL, FIRMS_CACHE_PATH

load_dotenv()
logger = logging.getLogger(__name__)

class FIRMSClient:
    def __init__(self, cache_path=FIRMS_CACHE_PATH):
        self.cache_path = cache_path
        self.api_key = os.getenv("FIRMS_MAP_KEY")
        
    def _read_cache(self) -> dict:
        """Reads cached hotspots from local JSON file."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read FIRMS cache: {e}")
        return None

    def _write_cache(self, data: dict):
        """Saves successful hotspot response to local JSON cache."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write FIRMS cache: {e}")

    def query_hotspots(self, lookback_days: int = 1, force_refresh: bool = False, region_override: dict = None) -> dict:
        """
        Queries NASA FIRMS API for thermal hotspots in configured bounding box.
        Fails gracefully to local cache if offline or API key is missing.
        
        Args:
            lookback_days: Number of days to look back for hotspots.
            force_refresh: If True, bypasses cache.
            region_override: Optional dict with 'bbox' key to override the default region.
                             e.g., {"bbox": {"west": 73.0, "south": 16.5, "east": 74.5, "north": 19.5}}
        """
        bbox = region_override["bbox"] if region_override and "bbox" in region_override else REGION["bbox"]

        # If API key is missing, return cached data if available, else generate simulated mock data
        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_nasa_firms_map_key_here":
            logger.warning("NASA FIRMS API Key (FIRMS_MAP_KEY) is missing in environment variables.")
            cached = self._read_cache()
            if cached:
                logger.info("Using cached satellite hotspot data (offline mode).")
                cached["cached"] = True
                return cached
            
            # Generate simulated mock hotspots within the active bbox
            import random
            random.seed(42)  # For deterministic demo hotspots
            num_mock_points = 5
            points = []
            lat_center = (bbox["south"] + bbox["north"]) / 2.0
            lon_center = (bbox["west"] + bbox["east"]) / 2.0
            
            for _ in range(num_mock_points):
                points.append({
                    "lat": float(lat_center + random.uniform(-0.3, 0.3)),
                    "lon": float(lon_center + random.uniform(-0.3, 0.3)),
                    "confidence": float(random.uniform(0.4, 0.95))
                })
                
            mock_result = {
                "hotspot_detected": True,
                "hotspot_count": len(points),
                "max_confidence": max([p["confidence"] for p in points]),
                "points": points,
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": True,
                "warning": "NASA FIRMS API Key (FIRMS_MAP_KEY) not set. Running in Offline Simulation Mode with mock data."
            }
            # Cache it for offline reuse
            self._write_cache(mock_result)
            return mock_result

        # Setup date parameters
        today = datetime.date.today().strftime("%Y-%m-%d")
        bbox = region_override["bbox"] if region_override and "bbox" in region_override else REGION["bbox"]
        
        # We query VIIRS_SNPP_NRT (S-NPP Suomi VIIRS, 375m resolution, standard for hotspots)
        url = FIRMS_API_URL.format(
            api_key=self.api_key,
            source="VIIRS_SNPP_NRT",
            west=bbox["west"],
            south=bbox["south"],
            east=bbox["east"],
            north=bbox["north"],
            range=lookback_days,
            date=today
        )

        try:
            logger.info(f"Querying NASA FIRMS: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                # If error, try MODIS_NRT as a fallback source
                logger.warning(f"VIIRS query returned status code {response.status_code}. Trying MODIS fallback...")
                url_fallback = FIRMS_API_URL.format(
                    api_key=self.api_key,
                    source="MODIS_NRT",
                    west=bbox["west"],
                    south=bbox["south"],
                    east=bbox["east"],
                    north=bbox["north"],
                    range=lookback_days,
                    date=today
                )
                response = requests.get(url_fallback, timeout=10)
                
            if response.status_code != 200:
                raise requests.RequestException(f"FIRMS API HTTP Error {response.status_code}: {response.text}")

            content = response.text.strip()
            points = []
            
            # Parse CSV response
            if content and "\n" in content:
                lines = content.split("\n")
                header = [h.strip().lower() for h in lines[0].split(",")]
                
                # Check column indices
                try:
                    lat_idx = header.index("latitude")
                    lon_idx = header.index("longitude")
                    conf_idx = header.index("confidence") if "confidence" in header else -1
                except ValueError as e:
                    logger.error(f"Invalid columns in FIRMS CSV: {e}")
                    raise
                
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) <= max(lat_idx, lon_idx):
                        continue
                    
                    try:
                        lat = float(parts[lat_idx])
                        lon = float(parts[lon_idx])
                        
                        # Parse confidence
                        conf_val = 0.5 # Default nominal
                        if conf_idx != -1 and conf_idx < len(parts):
                            raw_conf = parts[conf_idx].strip().lower()
                            if raw_conf == 'h' or raw_conf == 'high':
                                conf_val = 0.9
                            elif raw_conf == 'n' or raw_conf == 'nominal':
                                conf_val = 0.5
                            elif raw_conf == 'l' or raw_conf == 'low':
                                conf_val = 0.2
                            else:
                                try:
                                    # Might be percentage (0-100)
                                    val = float(raw_conf)
                                    conf_val = val / 100.0 if val > 1.0 else val
                                except ValueError:
                                    pass
                                    
                        # Bounding box sanity check
                        if (bbox["west"] <= lon <= bbox["east"]) and (bbox["south"] <= lat <= bbox["north"]):
                            points.append({
                                "lat": lat,
                                "lon": lon,
                                "confidence": conf_val
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing FIRMS row '{line}': {e}")
                        
            # Deduplicate close points within Sahyadri belt to avoid alert fatigue (IEEE 7010 compliance)
            deduplicated_points = self._deduplicate_points(points)
            
            hotspot_detected = len(deduplicated_points) > 0
            max_confidence = max([p["confidence"] for p in deduplicated_points]) if hotspot_detected else 0.0
            
            result = {
                "hotspot_detected": hotspot_detected,
                "hotspot_count": len(deduplicated_points),
                "max_confidence": max_confidence,
                "points": deduplicated_points,
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": False
            }
            
            # Cache the successful result
            self._write_cache(result)
            return result

        except Exception as e:
            logger.error(f"NASA FIRMS API failure: {e}")
            # Try to read cache
            cached = self._read_cache()
            if cached:
                logger.info("Gracefully fell back to cached FIRMS hotspot data.")
                cached["cached"] = True
                return cached
            
            # If no cache available, return None for hotspot_detected to signal "unknown" to the fusion engine
            return {
                "hotspot_detected": None,
                "hotspot_count": 0,
                "max_confidence": 0.0,
                "points": [],
                "error": f"API error and no local cache available: {str(e)}"
            }
            
    def _deduplicate_points(self, points: list, grid_size: float = 0.01) -> list:
        """
        IEEE 7010: Deduplicates alerts that are geographically extremely close (within grid_size degrees)
        to prevent redundant notification storms (alert fatigue).
        """
        grid = {}
        for p in points:
            # Round coordinates to nearest grid cell
            lat_grid = round(p["lat"] / grid_size) * grid_size
            lon_grid = round(p["lon"] / grid_size) * grid_size
            key = (round(lat_grid, 4), round(lon_grid, 4))
            
            if key not in grid:
                grid[key] = p
            else:
                # Keep the hotspot with higher confidence
                if p["confidence"] > grid[key]["confidence"]:
                    grid[key] = p
        return list(grid.values())
