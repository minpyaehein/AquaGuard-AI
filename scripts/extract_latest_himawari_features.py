from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from satpy import Scene


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_himawari_download.json"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_thaton_himawari_features.json"
)


# ============================================================
# TARGET AND FEATURE SETTINGS
# ============================================================

BAND = "B13"
TARGET_NAME = "Thaton Target Point"
TARGET_LATITUDE = 16.91867
TARGET_LONGITUDE = 97.37001

# B13 is too coarse for street-level analysis. The nearest B13 pixel is
# used for the target value; neighborhood metrics use a 5 km radius.
CLOUD_NEIGHBORHOOD_RADIUS_KM = 5.0

# Prototype cloud thresholds. These require later calibration against
# verified heavy-rain and non-heavy-rain cases.
COLD_CLOUD_THRESHOLD_K = 240.0
CLOUD_COOLING_THRESHOLD_K = -4.0

EXPECTED_FRAME_INTERVAL_MINUTES = 10
MINIMUM_REQUIRED_FRAMES = 7


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_utc(value: str) -> datetime:
    return datetime.strptime(
        value,
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/himawari_aws.py --frames 7 --band B13 first."
        )

    return json.loads(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )


def validate_frame_files(frame: dict) -> list[str]:
    files = [Path(value) for value in frame.get("files", [])]

    if len(files) != 10:
        raise ValueError(
            f"Frame {frame.get('timestamp_utc')} has {len(files)} files; "
            "10 B13 segment files were expected."
        )

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Himawari segment files:\n" + "\n".join(missing)
        )

    return [str(path) for path in files]


def haversine_km(
    latitude_1: np.ndarray,
    longitude_1: np.ndarray,
    latitude_2: float,
    longitude_2: float,
) -> np.ndarray:
    earth_radius_km = 6371.0088

    lat1 = np.radians(latitude_1)
    lon1 = np.radians(longitude_1)
    lat2 = np.radians(latitude_2)
    lon2 = np.radians(longitude_2)

    delta_latitude = lat1 - lat2
    delta_longitude = lon1 - lon2

    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_longitude / 2.0) ** 2
    )

    return 2.0 * earth_radius_km * np.arcsin(
        np.minimum(1.0, np.sqrt(value))
    )


# ============================================================
# READ ONE HIMAWARI FRAME
# ============================================================

def read_frame(frame: dict) -> dict:
    timestamp = frame["timestamp_utc"]
    files = validate_frame_files(frame)

    print("Reading:", timestamp, "UTC")

    scene = Scene(
        filenames=files,
        reader="ahi_hsd",
    )
    scene.load([BAND])

    band = scene[BAND]
    data = np.asarray(band.values, dtype=np.float32)
    data[~np.isfinite(data)] = np.nan

    area = band.attrs.get("area")
    if area is None:
        raise ValueError(
            f"No geolocation area was found for {timestamp}."
        )

    # Full-disk geolocation arrays are used only to locate a small window
    # around Thaton. The nearest valid pixel is selected objectively.
    longitudes, latitudes = area.get_lonlats()
    longitudes = np.asarray(longitudes, dtype=np.float32)
    latitudes = np.asarray(latitudes, dtype=np.float32)

    valid_geo = (
        np.isfinite(longitudes)
        & np.isfinite(latitudes)
        & np.isfinite(data)
    )

    # First create a broad geographic subset around the target to avoid
    # selecting an unrelated pixel elsewhere on the full disk.
    broad = (
        valid_geo
        & (latitudes >= TARGET_LATITUDE - 0.20)
        & (latitudes <= TARGET_LATITUDE + 0.20)
        & (longitudes >= TARGET_LONGITUDE - 0.20)
        & (longitudes <= TARGET_LONGITUDE + 0.20)
    )

    if not np.any(broad):
        raise ValueError(
            f"No valid geolocated B13 pixels were found near the target "
            f"for {timestamp}."
        )

    candidate_rows, candidate_columns = np.where(broad)
    candidate_distances = haversine_km(
        latitudes[broad],
        longitudes[broad],
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
    )

    nearest_index = int(np.nanargmin(candidate_distances))
    target_row = int(candidate_rows[nearest_index])
    target_column = int(candidate_columns[nearest_index])
    nearest_distance_km = float(candidate_distances[nearest_index])

    # Build neighborhood metrics from all valid pixels within the requested
    # radius. If the radius contains none, include the nearest pixel.
    distances = haversine_km(
        latitudes,
        longitudes,
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
    )
    neighborhood_mask = (
        valid_geo
        & (distances <= CLOUD_NEIGHBORHOOD_RADIUS_KM)
    )

    if not np.any(neighborhood_mask):
        neighborhood_mask[target_row, target_column] = True

    neighborhood_values = data[neighborhood_mask]

    return {
        "timestamp": timestamp,
        "datetime": parse_utc(timestamp),
        "target_temperature_k": float(data[target_row, target_column]),
        "minimum_temperature_k": float(np.nanmin(neighborhood_values)),
        "mean_temperature_k": float(np.nanmean(neighborhood_values)),
        "neighborhood_values": neighborhood_values.astype(np.float32),
        "neighborhood_mask": neighborhood_mask,
        "data": data,
        "nearest_pixel_latitude": float(latitudes[target_row, target_column]),
        "nearest_pixel_longitude": float(longitudes[target_row, target_column]),
        "nearest_pixel_distance_km": nearest_distance_km,
        "neighborhood_pixel_count": int(neighborhood_values.size),
    }


# ============================================================
# TEMPORAL FEATURE HELPERS
# ============================================================

def find_reference_frame(
    records: list[dict],
    minutes_before_latest: int,
) -> dict | None:
    latest_time = records[-1]["datetime"]
    desired_seconds = minutes_before_latest * 60

    candidates = records[:-1]
    if not candidates:
        return None

    best = min(
        candidates,
        key=lambda record: abs(
            (latest_time - record["datetime"]).total_seconds()
            - desired_seconds
        ),
    )

    actual_seconds = (
        latest_time - best["datetime"]
    ).total_seconds()

    tolerance_seconds = EXPECTED_FRAME_INTERVAL_MINUTES * 60 / 2
    if abs(actual_seconds - desired_seconds) > tolerance_seconds:
        return None

    return best


def target_temperature_change(
    latest: dict,
    reference: dict | None,
) -> float | None:
    if reference is None:
        return None

    return round(
        latest["target_temperature_k"]
        - reference["target_temperature_k"],
        6,
    )


def neighborhood_growth_fraction(
    latest: dict,
    reference: dict | None,
) -> float | None:
    if reference is None:
        return None

    # Frames share the same fixed AHI grid. Use the latest neighborhood mask
    # to compare like-for-like pixels across time.
    mask = latest["neighborhood_mask"]
    latest_values = latest["data"][mask]
    reference_values = reference["data"][mask]

    valid = np.isfinite(latest_values) & np.isfinite(reference_values)
    if not np.any(valid):
        return None

    cooling = latest_values[valid] - reference_values[valid]
    return round(
        float(np.mean(cooling <= CLOUD_COOLING_THRESHOLD_K)),
        6,
    )


# ============================================================
# BUILD FINAL FEATURES
# ============================================================

def build_features(records: list[dict]) -> dict:
    latest = records[-1]
    reference_10m = find_reference_frame(records, 10)
    reference_30m = find_reference_frame(records, 30)
    reference_60m = find_reference_frame(records, 60)

    latest_neighborhood = latest["neighborhood_values"]
    cold_cloud_fraction = float(
        np.mean(latest_neighborhood <= COLD_CLOUD_THRESHOLD_K)
    )

    data_age_minutes = (
        utc_now() - latest["datetime"]
    ).total_seconds() / 60.0

    return {
        "source": "NOAA AWS Himawari-9 AHI-L1b-FLDK B13",
        "target": {
            "name": TARGET_NAME,
            "latitude": TARGET_LATITUDE,
            "longitude": TARGET_LONGITUDE,
            "cloud_neighborhood_radius_km": CLOUD_NEIGHBORHOOD_RADIUS_KM,
        },
        "latest_time_utc": latest["timestamp"],
        "data_age_minutes": round(float(data_age_minutes), 3),
        "frame_count": len(records),
        "frame_interval_minutes": EXPECTED_FRAME_INTERVAL_MINUTES,
        "target_temperature_k": round(
            latest["target_temperature_k"], 6
        ),
        "minimum_temperature_within_neighborhood_k": round(
            latest["minimum_temperature_k"], 6
        ),
        "mean_temperature_within_neighborhood_k": round(
            latest["mean_temperature_k"], 6
        ),
        "temperature_change_10m_k": target_temperature_change(
            latest, reference_10m
        ),
        "temperature_change_30m_k": target_temperature_change(
            latest, reference_30m
        ),
        "temperature_change_60m_k": target_temperature_change(
            latest, reference_60m
        ),
        "cold_cloud_fraction": round(cold_cloud_fraction, 6),
        "cloud_growth_fraction_60m": neighborhood_growth_fraction(
            latest, reference_60m
        ),
        "nearest_b13_pixel": {
            "latitude": round(latest["nearest_pixel_latitude"], 6),
            "longitude": round(latest["nearest_pixel_longitude"], 6),
            "distance_from_target_km": round(
                latest["nearest_pixel_distance_km"], 6
            ),
        },
        "neighborhood_pixel_count": latest["neighborhood_pixel_count"],
        "thresholds": {
            "cold_cloud_threshold_k": COLD_CLOUD_THRESHOLD_K,
            "cloud_cooling_threshold_k": CLOUD_COOLING_THRESHOLD_K,
        },
        "interpretation": {
            "negative_temperature_change": (
                "Cloud-top brightness temperature decreased; this is a "
                "prototype cloud-development indicator, not direct rainfall."
            ),
            "cold_cloud_fraction": (
                "Fraction of valid B13 pixels in the cloud neighborhood "
                "at or below the prototype cold-cloud threshold."
            ),
            "cloud_growth_fraction_60m": (
                "Fraction of valid neighborhood pixels that cooled by at "
                "least the prototype threshold during the previous hour."
            ),
        },
        "limitations": [
            "Himawari B13 observes infrared cloud-top brightness temperature, not ground flooding.",
            "B13 is coarse relative to street-level analysis; the nearest satellite pixel represents an area around the target.",
            "Cold-cloud and cooling thresholds are prototype values and require calibration against verified events.",
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    manifest = load_manifest()
    frames = manifest.get("frames", [])

    if len(frames) < MINIMUM_REQUIRED_FRAMES:
        raise ValueError(
            f"At least {MINIMUM_REQUIRED_FRAMES} frames are required for "
            "a 60-minute feature sequence. Run himawari_aws.py with "
            "--frames 7."
        )

    # Keep the newest seven frames and ensure chronological order.
    frames = sorted(
        frames,
        key=lambda frame: parse_utc(frame["timestamp_utc"]),
    )[-MINIMUM_REQUIRED_FRAMES:]

    records = [read_frame(frame) for frame in frames]
    features = build_features(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(features, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("LATEST HIMAWARI FEATURES CREATED")
    print("=" * 70)
    print("Latest time:", features["latest_time_utc"], "UTC")
    print("Target temperature:", features["target_temperature_k"], "K")
    print(
        "10-minute change:",
        features["temperature_change_10m_k"],
        "K",
    )
    print(
        "30-minute change:",
        features["temperature_change_30m_k"],
        "K",
    )
    print(
        "60-minute change:",
        features["temperature_change_60m_k"],
        "K",
    )
    print("Cold-cloud fraction:", features["cold_cloud_fraction"])
    print(
        "Cloud-growth fraction (60m):",
        features["cloud_growth_fraction_60m"],
    )
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
