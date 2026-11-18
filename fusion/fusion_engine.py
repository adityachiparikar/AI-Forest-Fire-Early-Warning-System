from config import FUSION_WEIGHTS, ALERT_THRESHOLD_WARNING, ALERT_THRESHOLD_CONFIRMED

def fuse_scores(weather_risk: dict, thermal: dict, image: dict, iot: dict) -> dict:
    """
    IEEE 7000: Fuses evidence from four independent sensor modalities.
    If a modality is missing, its weights are distributed proportionally.

    Returns:
    {
        "confidence_score": float 0-100,
        "alert_level": str (Watch / Warning / Confirmed Fire),
        "reasoning": dict
    }
    """
    # 1. Parse and extract normalized scores (0.0 to 1.0) and availability
    modalities = {}

    # Layer 1: Weather Risk
    if weather_risk and weather_risk.get("risk_score") is not None and "error" not in weather_risk:
        modalities["weather"] = {
            "score": float(weather_risk["risk_score"]),
            "original_weight": FUSION_WEIGHTS["weather"],
            "raw_info": f"Weather Risk: {weather_risk.get('risk_level')}"
        }

    # Layer 2: Thermal Satellite
    if (thermal and
        thermal.get("hotspot_detected") is not None and
        "error" not in thermal):

        # If hotspot detected, score is its max confidence. If not, score is 0.0
        score = float(thermal["max_confidence"]) if thermal["hotspot_detected"] else 0.0
        modalities["thermal"] = {
            "score": score,
            "original_weight": FUSION_WEIGHTS["thermal"],
            "raw_info": f"Satellite Hotspots: {thermal.get('hotspot_count', 0)} detected"
        }

    # Layer 3: Vision
    if image and image.get("confidence") is not None and "error" not in image:
        modalities["image"] = {
            "score": float(image["confidence"]),
            "original_weight": FUSION_WEIGHTS["image"],
            "raw_info": f"Vision Fire Confidence: {image.get('confidence', 0.0):.2%}"
        }

    # Layer 4: IoT Ground Sensors
    if iot and iot.get("iot_confidence") is not None and "error" not in iot and not iot.get("stale", False):
        modalities["iot"] = {
            "score": float(iot["iot_confidence"]),
            "original_weight": FUSION_WEIGHTS["iot"],
            "raw_info": f"IoT Anomaly: {'Yes' if iot.get('anomaly_flag') else 'No'}"
        }

    # 2. Distribute weights proportionally if layers are missing
    if not modalities:
        # No layers available at all, return safe default watch
        return {
            "confidence_score": 0.0,
            "alert_level": "Watch",
            "reasoning": {"error": "All evidence layers are offline or missing data"}
        }

    total_original_weight = sum([m["original_weight"] for m in modalities.values()])

    # Calculate weighted fusion score
    weighted_sum = 0.0
    contributions = {}

    for name, info in modalities.items():
        # Redistributed weight
        adjusted_weight = info["original_weight"] / total_original_weight
        contribution = info["score"] * adjusted_weight
        weighted_sum += contribution

        contributions[name] = {
            "score_percentage": info["score"] * 100.0,
            "effective_weight": adjusted_weight,
            "contribution_points": contribution * 100.0,
            "status": info["raw_info"]
        }

    # Standardize final score to 0 - 100 scale
    final_score = round(weighted_sum * 100, 1)

    # 3. Apply Asymmetric Risk Alert Tiers (IEEE 7000 Ethical Compliance)
    # The costs of false negatives (undetected forest fires) are catastrophically
    # higher than false positives (temporary false alarms). Thus, thresholds are
    # biased toward caution:
    # 0 - 40% Watch
    # Wait, the comment got cut off. Let me rewrite the comment properly.
    # 0 - 40% Watch
    # 41 - 70% Warning (elevated caution, deployment of ground checks)
    # 71 - 100% Confirmed Fire (immediate automated alert propagation)
    thresholds = [(ALERT_THRESHOLD_CONFIRMED, "Confirmed Fire"), (ALERT_THRESHOLD_WARNING, "Warning")]
    for threshold, label in thresholds:
        if final_score >= threshold:
            alert_level = label
            break
    else:
        alert_level = "Watch"

    reasoning = {
        "active_layers": list(modalities.keys()),
        "offline_layers": [k for k in FUSION_WEIGHTS.keys() if k not in modalities],
        "contributions": contributions,
        "weight_redistributed": total_original_weight < 1.0
    }

    return {
        "confidence_score": final_score,
        "alert_level": alert_level,
        "reasoning": reasoning
    }