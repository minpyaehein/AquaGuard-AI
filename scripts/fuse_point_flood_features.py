from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POINT_RESULT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latest_thaton_point_flood_detection.json"
)

HIMAWARI_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_thaton_himawari_features.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latest_thaton_multisource_flood_nowcast.json"
)

# Prototype evidence thresholds. These are not scientifically calibrated.
MODERATE_TARGET_TEMPERATURE_K = 250.0
MODERATE_COOLING_10M_K = -4.0
MODERATE_COOLING_30M_K = -4.0
MODERATE_COLD_CLOUD_FRACTION = 0.10
MODERATE_CLOUD_GROWTH_FRACTION = 0.10

HIGH_TARGET_TEMPERATURE_K = 240.0
HIGH_COOLING_30M_K = -8.0
HIGH_COLD_CLOUD_FRACTION = 0.25
HIGH_CLOUD_GROWTH_FRACTION = 0.25

MAX_HIMAWARI_AGE_MINUTES = 90.0


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def less_or_equal(value, threshold):
    value = numeric(value)
    return value is not None and value <= threshold


def greater_or_equal(value, threshold):
    value = numeric(value)
    return value is not None and value >= threshold


def determine_himawari_level(features: dict) -> dict:
    age_minutes = numeric(features.get("data_age_minutes"))

    if age_minutes is None or age_minutes < 0:
        freshness = "UNKNOWN"
    elif age_minutes <= 30:
        freshness = "CURRENT"
    elif age_minutes <= MAX_HIMAWARI_AGE_MINUTES:
        freshness = "RECENT"
    else:
        freshness = "STALE"

    target_temperature = features.get("target_temperature_k")
    change_10m = features.get("temperature_change_10m_k")
    change_30m = features.get("temperature_change_30m_k")
    change_60m = features.get("temperature_change_60m_k")
    cold_fraction = features.get("cold_cloud_fraction")
    growth_fraction = features.get("cloud_growth_fraction_60m")

    high_signals = {
        "very_cold_target_pixel": less_or_equal(
            target_temperature,
            HIGH_TARGET_TEMPERATURE_K,
        ),
        "strong_30m_cooling": less_or_equal(
            change_30m,
            HIGH_COOLING_30M_K,
        ),
        "large_cold_cloud_fraction": greater_or_equal(
            cold_fraction,
            HIGH_COLD_CLOUD_FRACTION,
        ),
        "large_cloud_growth_fraction": greater_or_equal(
            growth_fraction,
            HIGH_CLOUD_GROWTH_FRACTION,
        ),
    }

    moderate_signals = {
        "cold_target_pixel": less_or_equal(
            target_temperature,
            MODERATE_TARGET_TEMPERATURE_K,
        ),
        "rapid_10m_cooling": less_or_equal(
            change_10m,
            MODERATE_COOLING_10M_K,
        ),
        "moderate_30m_cooling": less_or_equal(
            change_30m,
            MODERATE_COOLING_30M_K,
        ),
        "cold_cloud_fraction_present": greater_or_equal(
            cold_fraction,
            MODERATE_COLD_CLOUD_FRACTION,
        ),
        "cloud_growth_present": greater_or_equal(
            growth_fraction,
            MODERATE_CLOUD_GROWTH_FRACTION,
        ),
    }

    high_count = sum(high_signals.values())
    moderate_count = sum(moderate_signals.values())

    if freshness == "STALE":
        level = "STALE"
    elif high_count >= 2:
        level = "HIGH"
    elif high_count >= 1 or moderate_count >= 1:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "level": level,
        "freshness": freshness,
        "high_signal_count": high_count,
        "moderate_signal_count": moderate_count,
        "high_signals": high_signals,
        "moderate_signals": moderate_signals,
        "note": (
            "Himawari B13 measures cloud-top brightness temperature. "
            "This evidence describes cloud development, not ground flooding."
        ),
    }


def determine_fused_status(
    point_result: dict,
    himawari_evidence: dict,
) -> tuple[str, str, list[str]]:
    original_status = str(
        point_result.get("status", "UNKNOWN")
    )
    original_confidence = str(
        point_result.get("confidence", "PRELIMINARY")
    )

    rainfall_level = str(
        point_result
        .get("evidence_levels", {})
        .get("rainfall", "UNAVAILABLE")
    )
    soil_level = str(
        point_result
        .get("evidence_levels", {})
        .get("soil_wetness", "UNAVAILABLE")
    )
    cloud_level = himawari_evidence["level"]

    reasons = [
        f"Original point-detector status: {original_status}",
        f"Rainfall evidence: {rainfall_level}",
        f"Soil-wetness evidence: {soil_level}",
        f"Himawari cloud-development evidence: {cloud_level}",
    ]

    # Sentinel-1 surface evidence remains authoritative for surface-water
    # status. Himawari never creates a surface-flood confirmation by itself.
    if original_status in {
        "POSSIBLE_CURRENT_SURFACE_FLOOD",
        "POSSIBLE_CURRENT_SURFACE_WATER_CHANGE",
        "RECENT_SURFACE_FLOOD_EVIDENCE",
    }:
        if cloud_level in {"MODERATE", "HIGH"}:
            reasons.append(
                "Recent cloud-development evidence supports the weather context."
            )
        return original_status, original_confidence, reasons

    if (
        cloud_level == "HIGH"
        and rainfall_level in {"MODERATE", "HIGH"}
        and soil_level in {"MODERATE", "HIGH"}
    ):
        reasons.append(
            "High cloud-development evidence coincides with elevated rainfall "
            "and wet-soil evidence."
        )
        return "HIGH_POINT_FLOOD_RISK", "PRELIMINARY", reasons

    if (
        cloud_level in {"MODERATE", "HIGH"}
        and rainfall_level in {"MODERATE", "HIGH"}
    ):
        reasons.append(
            "Cloud-development and rainfall evidence are both elevated."
        )
        return "MODERATE_POINT_FLOOD_RISK", "PRELIMINARY", reasons

    if original_status in {
        "HIGH_FLOOD_RISK",
        "MODERATE_FLOOD_RISK",
    }:
        return original_status, original_confidence, reasons

    reasons.append(
        "The combined evidence does not meet the prototype elevated-risk rules."
    )
    return "LOW_CURRENT_POINT_FLOOD_EVIDENCE", "PRELIMINARY", reasons


def main():
    point_result = read_json(POINT_RESULT_FILE)
    himawari_features = read_json(HIMAWARI_FEATURE_FILE)

    himawari_evidence = determine_himawari_level(
        himawari_features
    )

    status, confidence, reasons = determine_fused_status(
        point_result,
        himawari_evidence,
    )

    output = {
        "location": point_result.get("location"),
        "target": point_result.get("target"),
        "prediction_type": (
            "POINT-BASED MULTISOURCE NEAR-REAL-TIME NOWCAST"
        ),
        "analysis_time_utc": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": status,
        "confidence": confidence,
        "evidence_levels": {
            **point_result.get("evidence_levels", {}),
            "himawari_cloud_development": (
                himawari_evidence["level"]
            ),
            "himawari_freshness": (
                himawari_evidence["freshness"]
            ),
        },
        "himawari": {
            "features": himawari_features,
            "evidence": himawari_evidence,
        },
        "gpm": point_result.get("gpm"),
        "soil_moisture": point_result.get("soil_moisture"),
        "gis_features": point_result.get("gis_features"),
        "sentinel1": point_result.get("sentinel1"),
        "decision_reasons": reasons,
        "method": (
            "Fusion of the existing point detector with latest Himawari-9 "
            "B13 cloud-temperature and cloud-development features."
        ),
        "limitations": [
            "This is a near-real-time nowcast, not a future flood forecast.",
            "Himawari cloud evidence does not directly confirm rainfall or ground flooding.",
            "GPM and SMAP are coarse regional inputs, not street-level measurements.",
            "All thresholds are prototype values and require validation against verified flood and non-flood events.",
        ],
        "warning": (
            "Research prototype only. Not an official public warning."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("THATON MULTISOURCE POINT FLOOD NOWCAST")
    print("=" * 72)
    print("Status:", status)
    print("Confidence:", confidence)
    print(
        "Himawari cloud evidence:",
        himawari_evidence["level"],
    )
    print(
        "Himawari freshness:",
        himawari_evidence["freshness"],
    )
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
