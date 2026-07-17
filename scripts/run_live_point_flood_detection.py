from datetime import datetime, timezone
from pathlib import Path
import json

import ee


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ID = "geoai-thaton-flood"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "latest_thaton_point_flood_detection.json"

# Target GPS point inside Thaton.
# Change these values when you want to assess another location.
TARGET_NAME = "Thaton Target Point"
TARGET_LATITUDE = 16.91867
TARGET_LONGITUDE = 97.37001

# The detector evaluates the surrounding neighborhood rather than a
# zero-area point. Sentinel-1 and terrain features are summarized inside
# this buffer. GPM and SMAP remain coarse regional satellite/model inputs.
ANALYSIS_RADIUS_METERS = 1000

# Sentinel-1 prototype detection thresholds
MAX_EVENT_VH_DB = -17.0
MIN_VH_DECREASE_DB = -3.0
MAX_SLOPE_DEGREES = 5.0
PERMANENT_WATER_OCCURRENCE = 90
MIN_FLOOD_AREA_PERCENTAGE = 0.5

# Prototype evidence levels
MODERATE_RAIN_24H_MM = 20.0
HIGH_RAIN_24H_MM = 50.0
MODERATE_RAIN_3DAY_MM = 60.0
HIGH_RAIN_3DAY_MM = 120.0
MODERATE_SURFACE_WETNESS = 0.60
HIGH_SURFACE_WETNESS = 0.80

# Data freshness
CURRENT_SENTINEL_MAX_AGE_DAYS = 2.0
RECENT_SENTINEL_MAX_AGE_DAYS = 7.0


# ============================================================
# TIME HELPERS
# ============================================================

def current_utc_datetime():
    return datetime.now(timezone.utc)


def parse_ee_datetime(value):
    if value is None:
        return None

    return datetime.strptime(
        value,
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=timezone.utc)


def calculate_age_hours(observation_time):
    parsed = parse_ee_datetime(observation_time)

    if parsed is None:
        return None

    return (
        current_utc_datetime() - parsed
    ).total_seconds() / 3600.0


# ============================================================
# EARTH ENGINE / AOI
# ============================================================

def initialize_earth_engine():
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized.")
    print("Project:", PROJECT_ID)


def create_target_area():
    target_point = ee.Geometry.Point(
        [TARGET_LONGITUDE, TARGET_LATITUDE],
        proj="EPSG:4326",
    )

    analysis_area = target_point.buffer(
        distance=ANALYSIS_RADIUS_METERS,
        maxError=1,
    )

    return target_point, analysis_area


# ============================================================
# GPM RAINFALL
# ============================================================

def create_gpm_collection(aoi, start_time, end_time):
    return (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
        .filterBounds(aoi)
        .filterDate(start_time, end_time)
        .select("precipitation")
    )


def calculate_gpm_accumulation(aoi, end_time, hours):
    start_time = end_time.advance(-hours, "hour")

    collection = create_gpm_collection(
        aoi,
        start_time,
        end_time.advance(30, "minute"),
    )

    image_count = collection.size().getInfo()

    if image_count == 0:
        return {"rainfall_mm": None, "image_count": 0}

    # IMERG precipitation is a rate in mm/hour.
    # Each observation represents a 30-minute interval.
    accumulated = (
        collection
        .map(lambda image: image.multiply(0.5))
        .sum()
        .rename("rainfall")
    )

    value = (
        accumulated.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=11132,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("rainfall")
        .getInfo()
    )

    return {
        "rainfall_mm": float(value) if value is not None else 0.0,
        "image_count": int(image_count),
    }


def calculate_latest_gpm(aoi):
    now = ee.Date(current_utc_datetime().isoformat())

    collection = (
        create_gpm_collection(
            aoi,
            now.advance(-14, "day"),
            now,
        )
        .sort("system:time_start", False)
    )

    if collection.size().getInfo() == 0:
        return {
            "available": False,
            "latest_time": None,
            "data_age_hours": None,
            "rain_1h_mm": None,
            "rain_6h_mm": None,
            "rain_24h_mm": None,
            "rain_3day_mm": None,
            "rain_7day_mm": None,
            "image_counts": {},
        }

    latest_image = ee.Image(collection.first())
    latest_time = ee.Date(latest_image.get("system:time_start"))
    latest_time_text = latest_time.format(
        "YYYY-MM-dd HH:mm:ss"
    ).getInfo()

    windows = {
        "rain_1h": 1,
        "rain_6h": 6,
        "rain_24h": 24,
        "rain_3day": 72,
        "rain_7day": 168,
    }

    accumulations = {
        name: calculate_gpm_accumulation(aoi, latest_time, hours)
        for name, hours in windows.items()
    }

    return {
        "available": True,
        "latest_time": latest_time_text,
        "data_age_hours": round(
            calculate_age_hours(latest_time_text), 3
        ),
        "rain_1h_mm": round(
            accumulations["rain_1h"]["rainfall_mm"], 6
        ),
        "rain_6h_mm": round(
            accumulations["rain_6h"]["rainfall_mm"], 6
        ),
        "rain_24h_mm": round(
            accumulations["rain_24h"]["rainfall_mm"], 6
        ),
        "rain_3day_mm": round(
            accumulations["rain_3day"]["rainfall_mm"], 6
        ),
        "rain_7day_mm": round(
            accumulations["rain_7day"]["rainfall_mm"], 6
        ),
        "image_counts": {
            name: result["image_count"]
            for name, result in accumulations.items()
        },
    }


# ============================================================
# SMAP SOIL MOISTURE
# ============================================================

def calculate_latest_soil_moisture(aoi):
    now = ee.Date(current_utc_datetime().isoformat())

    collection = (
        ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
        .filterBounds(aoi)
        .filterDate(now.advance(-21, "day"), now)
        .select([
            "sm_surface",
            "sm_rootzone",
            "sm_surface_wetness",
        ])
        .sort("system:time_start", False)
    )

    if collection.size().getInfo() == 0:
        return {
            "available": False,
            "latest_time": None,
            "data_age_hours": None,
            "surface_soil_moisture": None,
            "rootzone_soil_moisture": None,
            "surface_wetness": None,
        }

    latest = ee.Image(collection.first())
    latest_time = (
        ee.Date(latest.get("system:time_start"))
        .format("YYYY-MM-dd HH:mm:ss")
        .getInfo()
    )

    values = latest.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=11000,
        maxPixels=1_000_000_000,
        bestEffort=True,
    ).getInfo()

    def optional_float(name):
        value = values.get(name)
        return round(float(value), 6) if value is not None else None

    return {
        "available": True,
        "latest_time": latest_time,
        "data_age_hours": round(
            calculate_age_hours(latest_time), 3
        ),
        "surface_soil_moisture": optional_float("sm_surface"),
        "rootzone_soil_moisture": optional_float("sm_rootzone"),
        "surface_wetness": optional_float("sm_surface_wetness"),
    }


# ============================================================
# STATIC GIS FEATURES AT THE TARGET POINT
# ============================================================

def calculate_target_gis_features(target_point, analysis_area):
    dem_collection = ee.ImageCollection(
        "COPERNICUS/DEM/GLO30_2024_1"
    )
    native_projection = (
        ee.Image(dem_collection.first())
        .select("DEM")
        .projection()
    )
    dem = (
        dem_collection
        .select("DEM")
        .mosaic()
        .setDefaultProjection(native_projection)
        .rename("elevation")
    )
    slope = ee.Terrain.slope(dem).rename("slope")

    water_occurrence = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)
        .rename("water_occurrence")
    )

    point_values = (
        dem.addBands(slope)
        .addBands(water_occurrence)
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=target_point,
            scale=30,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    buffer_values = (
        dem.addBands(slope)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=analysis_area,
            scale=30,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .getInfo()
    )

    def number(values, name):
        value = values.get(name)
        return round(float(value), 6) if value is not None else None

    occurrence = number(point_values, "water_occurrence")

    return {
        "elevation_at_point_m": number(point_values, "elevation"),
        "slope_at_point_degrees": number(point_values, "slope"),
        "permanent_water_occurrence_percent": occurrence,
        "point_is_historically_permanent_water": bool(
            occurrence is not None
            and occurrence >= PERMANENT_WATER_OCCURRENCE
        ),
        "mean_elevation_within_buffer_m": number(
            buffer_values, "elevation"
        ),
        "mean_slope_within_buffer_degrees": number(
            buffer_values, "slope"
        ),
    }


# ============================================================
# SENTINEL-1 LATEST SAME-ORBIT PAIR
# ============================================================

def create_sentinel1_collection(aoi):
    now = ee.Date(current_utc_datetime().isoformat())

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(now.advance(-180, "day"), now)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(["VV", "VH"])
        .sort("system:time_start", False)
    )


def get_latest_sentinel1_pair(aoi):
    collection = create_sentinel1_collection(aoi)

    if collection.size().getInfo() == 0:
        return None

    latest_first = ee.Image(collection.first())
    latest_time = ee.Date(latest_first.get("system:time_start"))
    latest_day_start = ee.Date(latest_time.format("YYYY-MM-dd"))

    orbit_pass = latest_first.get(
        "orbitProperties_pass"
    ).getInfo()
    relative_orbit = latest_first.get(
        "relativeOrbitNumber_start"
    ).getInfo()
    platform = latest_first.get("platform_number").getInfo()

    same_orbit = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(
            ee.Filter.eq(
                "relativeOrbitNumber_start", relative_orbit
            )
        )
        .filter(ee.Filter.eq("platform_number", platform))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(["VV", "VH"])
    )

    latest_day_collection = same_orbit.filterDate(
        latest_day_start,
        latest_day_start.advance(1, "day"),
    )

    latest_image = latest_day_collection.mosaic().clip(aoi)

    previous_collection = (
        same_orbit
        .filterDate(
            latest_day_start.advance(-120, "day"),
            latest_day_start,
        )
        .sort("system:time_start", False)
    )

    if previous_collection.size().getInfo() == 0:
        return None

    previous_first = ee.Image(previous_collection.first())
    previous_time = ee.Date(
        previous_first.get("system:time_start")
    )
    previous_day_start = ee.Date(
        previous_time.format("YYYY-MM-dd")
    )

    previous_day_collection = same_orbit.filterDate(
        previous_day_start,
        previous_day_start.advance(1, "day"),
    )

    previous_image = previous_day_collection.mosaic().clip(aoi)

    latest_time_text = latest_time.format(
        "YYYY-MM-dd HH:mm:ss"
    ).getInfo()
    previous_time_text = previous_time.format(
        "YYYY-MM-dd HH:mm:ss"
    ).getInfo()

    return {
        "latest_image": latest_image,
        "previous_image": previous_image,
        "latest_time": latest_time_text,
        "previous_time": previous_time_text,
        "latest_data_age_hours": round(
            calculate_age_hours(latest_time_text), 3
        ),
        "latest_scene_count": latest_day_collection.size().getInfo(),
        "previous_scene_count": previous_day_collection.size().getInfo(),
        "orbit_pass": orbit_pass,
        "relative_orbit": relative_orbit,
        "platform": platform,
    }


# ============================================================
# SENTINEL-1 FLOOD DETECTION
# ============================================================

def detect_sentinel1_flood(aoi, pair):
    latest_vh = pair["latest_image"].select("VH")
    previous_vh = pair["previous_image"].select("VH")

    vh_change = latest_vh.subtract(previous_vh)

    current_water_candidate = latest_vh.lt(MAX_EVENT_VH_DB)
    backscatter_decrease = vh_change.lt(MIN_VH_DECREASE_DB)

    dem_collection = ee.ImageCollection(
        "COPERNICUS/DEM/GLO30_2024_1"
    )
    native_projection = (
        ee.Image(dem_collection.first())
        .select("DEM")
        .projection()
    )
    dem = (
        dem_collection
        .select("DEM")
        .mosaic()
        .setDefaultProjection(native_projection)
        .clip(aoi)
    )
    low_slope = ee.Terrain.slope(dem).lt(MAX_SLOPE_DEGREES)

    permanent_water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .gte(PERMANENT_WATER_OCCURRENCE)
    )

    probable_flood = (
        current_water_candidate
        .And(backscatter_decrease)
        .And(low_slope)
        .And(permanent_water.Not())
        .selfMask()
        .rename("probable_flood")
    )

    flood_area_value = (
        probable_flood
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("probable_flood")
        .getInfo()
    )

    total_area = aoi.area(maxError=1).getInfo()
    flood_area_m2 = float(flood_area_value or 0.0)
    flood_percentage = flood_area_m2 / float(total_area) * 100.0

    return {
        "detected": flood_percentage >= MIN_FLOOD_AREA_PERCENTAGE,
        "probable_flood_area_m2": round(flood_area_m2, 3),
        "probable_flood_area_km2": round(
            flood_area_m2 / 1_000_000, 6
        ),
        "probable_flood_percentage": round(
            flood_percentage, 6
        ),
    }


# ============================================================
# EVIDENCE / STATUS LOGIC
# ============================================================

def determine_rainfall_level(gpm):
    if not gpm["available"]:
        return "UNAVAILABLE"

    if (
        gpm["rain_24h_mm"] >= HIGH_RAIN_24H_MM
        or gpm["rain_3day_mm"] >= HIGH_RAIN_3DAY_MM
    ):
        return "HIGH"

    if (
        gpm["rain_24h_mm"] >= MODERATE_RAIN_24H_MM
        or gpm["rain_3day_mm"] >= MODERATE_RAIN_3DAY_MM
    ):
        return "MODERATE"

    return "LOW"


def determine_soil_wetness_level(soil):
    if not soil["available"] or soil["surface_wetness"] is None:
        return "UNAVAILABLE"

    if soil["surface_wetness"] >= HIGH_SURFACE_WETNESS:
        return "HIGH"

    if soil["surface_wetness"] >= MODERATE_SURFACE_WETNESS:
        return "MODERATE"

    return "LOW"


def determine_sentinel_freshness(sentinel):
    if not sentinel.get("available"):
        return "UNAVAILABLE"

    age_hours = sentinel.get("data_age_hours")

    if age_hours is None:
        return "UNKNOWN"

    age_days = age_hours / 24.0

    if age_days <= CURRENT_SENTINEL_MAX_AGE_DAYS:
        return "CURRENT"

    if age_days <= RECENT_SENTINEL_MAX_AGE_DAYS:
        return "RECENT"

    return "STALE"


def determine_final_status(gpm, soil, sentinel):
    rainfall_level = determine_rainfall_level(gpm)
    soil_level = determine_soil_wetness_level(soil)
    sentinel_freshness = determine_sentinel_freshness(sentinel)
    sentinel_detected = bool(sentinel.get("detected", False))

    if sentinel_detected and sentinel_freshness == "CURRENT":
        if (
            rainfall_level in ("MODERATE", "HIGH")
            or soil_level in ("MODERATE", "HIGH")
        ):
            status = "POSSIBLE_CURRENT_SURFACE_FLOOD"
            confidence = "MODERATE"
        else:
            status = "POSSIBLE_CURRENT_SURFACE_WATER_CHANGE"
            confidence = "PRELIMINARY"

    elif sentinel_detected and sentinel_freshness == "RECENT":
        status = "RECENT_SURFACE_FLOOD_EVIDENCE"
        confidence = "PRELIMINARY"

    elif rainfall_level == "HIGH" and soil_level in (
        "MODERATE",
        "HIGH",
    ):
        status = "HIGH_FLOOD_RISK"
        confidence = "PRELIMINARY"

    elif rainfall_level in ("MODERATE", "HIGH") or soil_level == "HIGH":
        status = "MODERATE_FLOOD_RISK"
        confidence = "PRELIMINARY"

    else:
        status = "LOW_CURRENT_FLOOD_EVIDENCE"
        confidence = "PRELIMINARY"

    return {
        "status": status,
        "confidence": confidence,
        "rainfall_level": rainfall_level,
        "soil_wetness_level": soil_level,
        "sentinel_freshness": sentinel_freshness,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    initialize_earth_engine()
    target_point, aoi = create_target_area()

    analysis_time = current_utc_datetime().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print("\nReading static GIS features at the target point...")
    gis_features = calculate_target_gis_features(
        target_point,
        aoi,
    )
    print("Elevation at point:", gis_features["elevation_at_point_m"], "m")
    print("Slope at point:", gis_features["slope_at_point_degrees"], "degrees")

    print("\nChecking latest available GPM rainfall...")
    gpm = calculate_latest_gpm(aoi)
    print("Latest GPM time:", gpm["latest_time"])
    print("Rainfall 1h:", gpm["rain_1h_mm"], "mm")
    print("Rainfall 6h:", gpm["rain_6h_mm"], "mm")
    print("Rainfall 24h:", gpm["rain_24h_mm"], "mm")
    print("Rainfall 3-day:", gpm["rain_3day_mm"], "mm")
    print("Rainfall 7-day:", gpm["rain_7day_mm"], "mm")

    print("\nChecking latest available SMAP soil moisture...")
    soil = calculate_latest_soil_moisture(aoi)
    print("Latest SMAP time:", soil["latest_time"])
    print("Surface soil moisture:", soil["surface_soil_moisture"])
    print("Root-zone soil moisture:", soil["rootzone_soil_moisture"])
    print("Surface wetness:", soil["surface_wetness"])

    print("\nChecking latest available Sentinel-1 data...")
    pair = get_latest_sentinel1_pair(aoi)

    sentinel = {
        "available": False,
        "latest_time": None,
        "previous_time": None,
        "data_age_hours": None,
        "detected": False,
    }

    if pair is not None:
        detection = detect_sentinel1_flood(aoi, pair)
        sentinel = {
            "available": True,
            "latest_time": pair["latest_time"],
            "previous_time": pair["previous_time"],
            "data_age_hours": pair["latest_data_age_hours"],
            "platform": pair["platform"],
            "orbit_pass": pair["orbit_pass"],
            "relative_orbit": pair["relative_orbit"],
            "latest_scene_count": pair["latest_scene_count"],
            "previous_scene_count": pair["previous_scene_count"],
            **detection,
        }

    decision = determine_final_status(gpm, soil, sentinel)

    result = {
        "location": "Thaton, Mon State, Myanmar",
        "target": {
            "name": TARGET_NAME,
            "latitude": TARGET_LATITUDE,
            "longitude": TARGET_LONGITUDE,
            "analysis_radius_m": ANALYSIS_RADIUS_METERS,
        },
        "prediction_type": "POINT-BASED NEAR-REAL-TIME NOWCAST",
        "analysis_time": analysis_time,
        "status": decision["status"],
        "confidence": decision["confidence"],
        "evidence_levels": {
            "rainfall": decision["rainfall_level"],
            "soil_wetness": decision["soil_wetness_level"],
            "sentinel1_freshness": decision["sentinel_freshness"],
        },
        "gis_features": gis_features,
        "gpm": gpm,
        "soil_moisture": soil,
        "sentinel1": sentinel,
        "method": (
            "Point-centered latest-available GPM rainfall, SMAP soil moisture, "
            "Sentinel-1 SAR change detection inside the local buffer, "
            "DEM terrain filtering and permanent-water removal"
        ),
        "limitations": [
            "This is a point-centered nowcast, not a future flood forecast.",
            "GPM and SMAP are coarse regional inputs and are not street-level measurements.",
            "Prototype thresholds have not yet been calibrated against multiple verified events.",
        ],
        "warning": (
            "Research prototype only. Not an official public warning."
        ),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("LATEST THATON POINT FLOOD NOWCAST")
    print("=" * 70)
    print("Target:", TARGET_NAME)
    print("Coordinates:", TARGET_LATITUDE, TARGET_LONGITUDE)
    print("Analysis radius:", ANALYSIS_RADIUS_METERS, "m")
    print("Analysis time:", analysis_time)
    print("Status:", result["status"])
    print("Confidence:", result["confidence"])
    print("Rainfall evidence:", result["evidence_levels"]["rainfall"])
    print(
        "Soil-wetness evidence:",
        result["evidence_levels"]["soil_wetness"],
    )
    print(
        "Sentinel-1 freshness:",
        result["evidence_levels"]["sentinel1_freshness"],
    )

    if sentinel["available"]:
        print("Latest Sentinel-1:", sentinel["latest_time"])
        print("Previous Sentinel-1:", sentinel["previous_time"])
        print("Sentinel-1 data age:", sentinel["data_age_hours"], "hours")
        print(
            "Probable flood area:",
            sentinel["probable_flood_area_km2"],
            "km²",
        )
        print(
            "Probable coverage:",
            sentinel["probable_flood_percentage"],
            "%",
        )
    else:
        print("Sentinel-1 surface confirmation unavailable.")

    print("\nOutput:", OUTPUT_JSON)
    print("IMPORTANT: Research prototype only. Not an official warning.")


if __name__ == "__main__":
    main()
