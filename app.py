"""
Multi-Modal Forest Fire Early-Warning System Dashboard
Streamlit Application Entry Point

Target Region: Western Maharashtra (Sahyadri Belt), India
IEEE Compliant: 7000, 7010, 2413, 1451
"""

import logging
import importlib
import streamlit as st
import pandas as pd
import folium
from dotenv import load_dotenv

# Pre-load TensorFlow on main thread to avoid DLL load issues inside worker threads on Windows
try:
    import tensorflow as tf
except ImportError:
    pass

from config import REGION, PRESET_REGIONS, FUSION_WEIGHTS, MODELS_DIR, IOT_SIM_CSV
from fusion.fusion_engine import fuse_scores

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Early-Warning System | Sahyadri Belt",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------------
# Helper: safely load layers
# ---------------------------------------------------------------------------
LAYER_CACHE_VERSION = "2026-07-18-ui-and-vision-fix"

@st.cache_resource
def _load(module, cls, cache_version):
    try:
        mod = importlib.import_module(module)
        mod = importlib.reload(mod)
        return getattr(mod, cls)()
    except Exception as e:
        logger.error(f"{cls} unavailable: {e}")
        return None

def _display_layer_name(layer_name):
    names = {
        "weather": "Weather",
        "thermal": "Thermal",
        "image": "Image",
        "iot": "IoT",
    }
    return names.get(str(layer_name).lower(), str(layer_name).title())


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Load all layer modules
    weather_predictor = _load('layer1_weather.predict', 'WeatherRiskPredictor', LAYER_CACHE_VERSION)
    firms_client = _load('layer2_thermal.firms_client', 'FIRMSClient', LAYER_CACHE_VERSION)
    vision_detector = _load('layer3_vision.detect', 'KerasVisionDetector', LAYER_CACHE_VERSION)
    iot_pipeline = _load('layer4_iot.sensor_pipeline', 'IoTSensorPipeline', LAYER_CACHE_VERSION)

    # ======================================================================
    # SIDEBAR: Region Selector & System Status
    # ======================================================================
    with st.sidebar:
        st.markdown("### Region Selector")
        region_names = list(PRESET_REGIONS.keys())
        selected_name = st.selectbox(
            "Select monitoring region",
            region_names,
            index=0,
            key="region_selector"
        )

        # Build the active region dict based on selection
        if selected_name == "Custom Location":
            st.markdown("#### Enter Custom Coordinates")
            custom_name = st.text_input("Location Name", value="My Custom Region", key="custom_name")
            cc1, cc2 = st.columns(2)
            with cc1:
                custom_lat = st.number_input("Center Latitude", value=18.0, min_value=-90.0, max_value=90.0, step=0.1, key="custom_lat")
                custom_west = st.number_input("West (lon)", value=73.0, min_value=-180.0, max_value=180.0, step=0.1, key="custom_west")
                custom_south = st.number_input("South (lat)", value=16.5, min_value=-90.0, max_value=90.0, step=0.1, key="custom_south")
            with cc2:
                custom_lon = st.number_input("Center Longitude", value=73.7, min_value=-180.0, max_value=180.0, step=0.1, key="custom_lon")
                custom_east = st.number_input("East (lon)", value=74.5, min_value=-180.0, max_value=180.0, step=0.1, key="custom_east")
                custom_north = st.number_input("North (lat)", value=19.5, min_value=-90.0, max_value=90.0, step=0.1, key="custom_north")
            active_region = {
                "name": custom_name,
                "bbox": {"west": custom_west, "south": custom_south, "east": custom_east, "north": custom_north},
                "center": {"lat": custom_lat, "lon": custom_lon}
            }
        else:
            preset = PRESET_REGIONS[selected_name]
            active_region = {
                "name": selected_name,
                "bbox": preset["bbox"],
                "center": preset["center"]
            }

        st.markdown(f"**Active:** {active_region['name']}")
        st.markdown(f"Bbox: `{active_region['bbox']['west']}, {active_region['bbox']['south']}, {active_region['bbox']['east']}, {active_region['bbox']['north']}`")
        st.markdown(f"Center: `{active_region['center']['lat']}, {active_region['center']['lon']}`")
        st.divider()

        st.markdown("### Layer Status")
        layers = {'Weather Risk': weather_predictor, 'Satellite Thermal': firms_client, 'Vision CNN': vision_detector, 'IoT Sensors': iot_pipeline}
        for name, obj in layers.items():
            status = "Ready" if obj else "Unavailable"
            status_color = "green" if obj else "red"
            st.markdown(f"**{name}:** :{status_color}[{status}]")
        st.divider()

        st.markdown("### Fusion Weights")
        st.table(pd.DataFrame({"Layer": [_display_layer_name(k) for k in FUSION_WEIGHTS], "Weight": [f"{v:.0%}" for v in FUSION_WEIGHTS.values()]}))

    # ---- Header ----
    st.title("Forest Fire Early-Warning System")
    st.caption(f"Multi-modal detection - {active_region['name']} - Real-time fusion of weather, satellite, imagery and IoT sensors")

    # ======================================================================
    # MAIN CONTENT: 3-column layout
    # ======================================================================

    # ---------- Row 1: Weather + Satellite Map ----------
    col_weather, col_map = st.columns([1, 1.5])

    # ---- Layer 1: Weather Risk Panel ----
    weather_result = None
    with col_weather:
        st.markdown("### Layer 1: Weather Risk Assessment")
        with st.expander("Configure Weather Parameters", expanded=True):
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                month = st.selectbox("Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], index=5)
                day = st.selectbox("Day", ['mon','tue','wed','thu','fri','sat','sun'], index=4)
                temp = st.slider("Temperature (C)", 0.0, 50.0, 30.0, 0.5)
                rh = st.slider("Relative Humidity (%)", 0, 100, 35)
                wind = st.slider("Wind Speed (km/h)", 0.0, 15.0, 5.0, 0.5)
            with w_col2:
                rain = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0, 0.1)
                ffmc = st.slider("FFMC", 18.0, 97.0, 86.0, 0.5)
                dmc = st.slider("DMC", 1.0, 300.0, 100.0, 1.0)
                dc = st.slider("DC", 7.0, 860.0, 400.0, 5.0)
                isi = st.slider("ISI", 0.0, 60.0, 10.0, 0.5)

            weather_input = {
                "month": month, "day": day, "FFMC": ffmc, "DMC": dmc,
                "DC": dc, "ISI": isi, "temp": temp, "RH": rh,
                "wind": wind, "rain": rain
            }

            if weather_predictor:
                weather_result = weather_predictor.predict(weather_input)
                st.metric("Weather Fire Risk", weather_result['risk_level'])
                st.caption(f"Risk score: {weather_result['risk_score']:.1%}")
            else:
                st.warning("Weather model not loaded. Run `python tools/generate_dummy_models.py` to generate model artifacts.")

    # ---- Layer 2: Satellite Thermal Map ----
    thermal_result = None
    with col_map:
        st.markdown("### Layer 2: Live FIRMS Satellite Hotspots")
        if firms_client:
            lookback = st.slider("Lookback Days", 1, 10, 1)
            if st.button("Refresh Satellite Data", key="btn_firms"):
                st.cache_data.clear()

            @st.cache_data(ttl=300)
            def fetch_hotspots(_lookback, _region_name, _region_bbox):
                return firms_client.query_hotspots(
                    lookback_days=_lookback,
                    region_override={"bbox": _region_bbox}
                )

            thermal_result = fetch_hotspots(lookback, active_region["name"], active_region["bbox"])

            # Build Folium Map
            m = folium.Map(
                location=[active_region["center"]["lat"], active_region["center"]["lon"]],
                zoom_start=8,
                tiles="OpenStreetMap"
            )
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Satellite Imagery"
            ).add_to(m)
            folium.LayerControl().add_to(m)

            # Draw bounding box rectangle
            bbox = active_region["bbox"]
            folium.Rectangle(
                bounds=[[bbox["south"], bbox["west"]], [bbox["north"], bbox["east"]]],
                color="#6366f1",
                weight=2,
                fill=True,
                fill_opacity=0.05,
                popup=active_region["name"]
            ).add_to(m)

            # Plot hotspots
            if thermal_result and thermal_result.get("points"):
                for pt in thermal_result["points"]:
                    conf = pt.get("confidence", 0.5)
                    color = "red" if conf > 0.7 else ("orange" if conf > 0.4 else "yellow")
                    folium.CircleMarker(
                        location=[pt["lat"], pt["lon"]],
                        radius=max(4, conf * 12),
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"Lat: {pt['lat']:.4f}, Lon: {pt['lon']:.4f}<br>Confidence: {conf:.0%}"
                    ).add_to(m)

            st.components.v1.html(m._repr_html_(), height=400)

            # Hotspot summary
            if thermal_result:
                if thermal_result.get("error"):
                    st.warning(f"Satellite data issue: {thermal_result['error']}")
                elif thermal_result.get("cached"):
                    st.info("Displaying cached satellite data (API offline or key not set).")

                count = thermal_result.get("hotspot_count", 0)
                max_conf = thermal_result.get("max_confidence", 0.0)
                src = "Cached" if thermal_result.get("cached") else "Live"
                st.metric("Thermal Hotspots", f"{count} Hotspot{'s' if count != 1 else ''}")
                st.caption(f"Max confidence: {max_conf:.0%} | Source: {src}")
        else:
            st.warning("FIRMS client not available.")
            m = folium.Map(location=[active_region["center"]["lat"], active_region["center"]["lon"]], zoom_start=8, tiles="OpenStreetMap")
            st.components.v1.html(m._repr_html_(), height=400)

    # ---------- Row 2: Vision + IoT ----------
    col_vision, col_iot = st.columns(2)

    # ---- Layer 3: Image Upload & Detection ----
    image_result = None
    with col_vision:
        st.markdown("### Layer 3: Fire Image Detection")
        uploaded_file = st.file_uploader("Upload a forest/fire image (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="img_upload")

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if vision_detector:
                with st.spinner("Analyzing image..."):
                    image_result = vision_detector.detect_fire(uploaded_file)

                if image_result.get("error"):
                    st.error(f"Detection error: {image_result['error']}")
                else:
                    conf = image_result["confidence"]
                    detected = image_result["fire_detected"]
                    label = "FIRE DETECTED" if detected else "No Fire Detected"
                    st.metric("Vision Analysis", label)
                    st.caption(f"Fire confidence: {conf:.1%}")
                    if image_result.get("warning"):
                        st.caption(f"Note: {image_result['warning']}")
            else:
                st.warning("Vision detector not loaded.")
        else:
            st.info("Upload an image to activate fire detection analysis.")

    # ---- Layer 4: IoT Sensor Stream ----
    iot_result = None
    with col_iot:
        st.markdown("### Layer 4: IoT Ground Sensor Stream")
        if iot_pipeline:
            # Simulated stream index from session state
            if "iot_index" not in st.session_state:
                st.session_state.iot_index = 0

            col_iot_ctrl1, col_iot_ctrl2 = st.columns(2)
            with col_iot_ctrl1:
                if st.button("Next Reading", key="btn_iot_next"):
                    st.session_state.iot_index += 1
            with col_iot_ctrl2:
                if st.button("Auto-play (10 readings)", key="btn_iot_auto"):
                    st.session_state.iot_index += 10

            # Read simulated sensor data
            reading = iot_pipeline.read_simulated_stream(st.session_state.iot_index)
            if reading:
                iot_result = iot_pipeline.analyze_telemetry(reading)

                # Show current readings as metrics
                channels = reading.get("channels", {})
                m1, m2, m3 = st.columns(3)
                with m1:
                    t_val = channels.get("temperature", {}).get("value", 0)
                    st.metric("Temperature", f"{t_val:.1f} C")
                with m2:
                    h_val = channels.get("humidity", {}).get("value", 0)
                    st.metric("Humidity", f"{h_val:.1f} %")
                with m3:
                    s_val = channels.get("smoke_concentration", {}).get("value", 0)
                    st.metric("Smoke (PPM)", f"{s_val:.0f}")

                # Plot historical data
                try:
                    df_sim = pd.read_csv(IOT_SIM_CSV)
                    end_idx = min(st.session_state.iot_index + 1, len(df_sim))
                    df_plot = df_sim.iloc[:end_idx]

                    chart_data = {
                        "Temperature (C)": df_plot["temp"].values,
                        "Smoke (PPM)": df_plot["smoke"].values / 10.0,  # scale for readability
                    }
                    st.area_chart(chart_data, height=200)
                except Exception:
                    pass

                # Show IoT result
                if iot_result:
                    anom = iot_result.get("anomaly_flag", False)
                    iot_conf = iot_result.get("iot_confidence", 0.0)
                    stale = iot_result.get("stale", False)
                    status_label = "ANOMALY" if anom else "Normal"
                    detail = f"Confidence: {iot_conf:.1%} | {'STALE DATA' if stale else 'Live'} | Reading #{st.session_state.iot_index}"
                    if anom:
                        st.error(f"{status_label}: {detail}")
                    else:
                        st.success(f"{status_label}: {detail}")
            else:
                st.warning("No sensor data available.")
        else:
            st.warning("IoT pipeline not loaded.")

    # ======================================================================
    # FUSION ALERT: The Visual Centerpiece
    # ======================================================================
    st.divider()
    st.markdown("## Fusion Alert Monitor")

    fusion_result = fuse_scores(
        weather_risk=weather_result,
        thermal=thermal_result,
        image=image_result,
        iot=iot_result
    )

    score = fusion_result["confidence_score"]
    alert = fusion_result["alert_level"]
    reasoning = fusion_result.get("reasoning", {})

    # Render alert
    col_alert, col_reasoning = st.columns([1, 1.5])

    with col_alert:
        if alert == "Confirmed Fire":
            st.error(f"**{alert}** | Fusion Score: **{score:.1f}%**")
        elif alert == "Warning":
            st.warning(f"**{alert}** | Fusion Score: **{score:.1f}%**")
        else:
            st.success(f"**{alert}** | Fusion Score: **{score:.1f}%**")

        st.metric("Fusion Confidence", f"{score:.1f}%")

        # Active/offline layers
        active = reasoning.get("active_layers", [])
        offline = reasoning.get("offline_layers", [])
        if active:
            st.markdown(f"**Active Layers:** {', '.join([_display_layer_name(l) for l in active])}")
        if offline:
            st.markdown(f"**Offline Layers:** {', '.join([_display_layer_name(l) for l in offline])}")
        if reasoning.get("weight_redistributed"):
            st.caption("Weights have been redistributed due to offline layers.")

    with col_reasoning:
        st.markdown("#### Per-Layer Reasoning Breakdown")
        contributions = reasoning.get("contributions", {})
        if contributions:
            rows = []
            for layer_name, info in contributions.items():
                rows.append({
                    "Layer": _display_layer_name(layer_name),
                    "Score": f"{info.get('score_percentage', 0):.1f}%",
                    "Weight": f"{info.get('effective_weight', 0):.0%}",
                    "Contribution": f"{info.get('contribution_points', 0):.1f} pts",
                    "Status": info.get("status", "")
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No layer data available for reasoning breakdown.")


if __name__ == "__main__":
    main()