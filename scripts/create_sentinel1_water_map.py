from __future__ import annotations

import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import ee
import folium


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ID = "geoai-thaton-flood"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "latest_thaton_flood_monitor.html"
OUTPUT_JSON = OUTPUT_DIR / "latest_thaton_water_map_summary.json"

TARGET_NAME = "Thaton Target Point"
TARGET_LATITUDE = 16.91867
TARGET_LONGITUDE = 97.37001
ANALYSIS_RADIUS_METERS = 1000

MAX_EVENT_VH_DB = -17.0
MIN_VH_DECREASE_DB = -3.0
MAX_SLOPE_DEGREES = 5.0
PERMANENT_WATER_OCCURRENCE = 90

# Search far enough back to find a same-orbit reference observation.
LATEST_SEARCH_DAYS = 180
REFERENCE_SEARCH_DAYS = 120


# ============================================================
# EARTH ENGINE HELPERS
# ============================================================

def initialize_earth_engine() -> None:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized.")
    print("Project:", PROJECT_ID)


def create_target_geometries():
    target = ee.Geometry.Point(
        [TARGET_LONGITUDE, TARGET_LATITUDE],
        proj="EPSG:4326",
    )
    analysis_area = target.buffer(
        ANALYSIS_RADIUS_METERS,
        maxError=1,
    )
    return target, analysis_area


def base_sentinel_collection(aoi):
    now = ee.Date(datetime.now(timezone.utc).isoformat())

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(now.advance(-LATEST_SEARCH_DAYS, "day"), now)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV",
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH",
            )
        )
        .select(["VV", "VH"])
        .sort("system:time_start", False)
    )


def get_latest_same_orbit_pair(aoi) -> dict:
    collection = base_sentinel_collection(aoi)

    if collection.size().getInfo() == 0:
        raise RuntimeError(
            "No VV/VH Sentinel-1 observations were found near the target."
        )

    latest_first = ee.Image(collection.first())
    latest_time = ee.Date(latest_first.get("system:time_start"))
    latest_day = ee.Date(latest_time.format("YYYY-MM-dd"))

    orbit_pass = latest_first.get("orbitProperties_pass").getInfo()
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
                "relativeOrbitNumber_start",
                relative_orbit,
            )
        )
        .filter(ee.Filter.eq("platform_number", platform))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV",
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH",
            )
        )
        .select(["VV", "VH"])
    )

    latest_day_collection = same_orbit.filterDate(
        latest_day,
        latest_day.advance(1, "day"),
    )
    latest_image = latest_day_collection.mosaic().clip(aoi)

    previous_collection = (
        same_orbit
        .filterDate(
            latest_day.advance(-REFERENCE_SEARCH_DAYS, "day"),
            latest_day,
        )
        .sort("system:time_start", False)
    )

    if previous_collection.size().getInfo() == 0:
        raise RuntimeError(
            "No previous same-orbit Sentinel-1 reference observation was found."
        )

    previous_first = ee.Image(previous_collection.first())
    previous_time = ee.Date(previous_first.get("system:time_start"))
    previous_day = ee.Date(previous_time.format("YYYY-MM-dd"))

    previous_day_collection = same_orbit.filterDate(
        previous_day,
        previous_day.advance(1, "day"),
    )
    previous_image = previous_day_collection.mosaic().clip(aoi)

    return {
        "latest_image": latest_image,
        "previous_image": previous_image,
        "latest_time": latest_time.format(
            "YYYY-MM-dd HH:mm:ss"
        ).getInfo(),
        "previous_time": previous_time.format(
            "YYYY-MM-dd HH:mm:ss"
        ).getInfo(),
        "platform": platform,
        "orbit_pass": orbit_pass,
        "relative_orbit": relative_orbit,
        "latest_scene_count": latest_day_collection.size().getInfo(),
        "previous_scene_count": previous_day_collection.size().getInfo(),
    }


# ============================================================
# WATER LAYERS
# ============================================================

def build_water_layers(aoi, pair: dict) -> dict:
    latest_vh = pair["latest_image"].select("VH")
    previous_vh = pair["previous_image"].select("VH")
    vh_change = latest_vh.subtract(previous_vh)

    current_dark_water = latest_vh.lt(MAX_EVENT_VH_DB)
    strong_decrease = vh_change.lt(MIN_VH_DECREASE_DB)

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

    occurrence = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .clip(aoi)
    )
    permanent_water = occurrence.gte(
        PERMANENT_WATER_OCCURRENCE
    ).selfMask().rename("permanent_water")

    probable_new_water = (
        current_dark_water
        .And(strong_decrease)
        .And(low_slope)
        .And(permanent_water.unmask(0).Not())
        .selfMask()
        .rename("probable_new_water")
    )

    return {
        "latest_vh": latest_vh,
        "vh_change": vh_change,
        "permanent_water": permanent_water,
        "probable_new_water": probable_new_water,
    }


def calculate_area_summary(aoi, layers: dict) -> dict:
    pixel_area = ee.Image.pixelArea()

    permanent_area = (
        layers["permanent_water"]
        .multiply(pixel_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("permanent_water")
        .getInfo()
    )

    new_water_area = (
        layers["probable_new_water"]
        .multiply(pixel_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("probable_new_water")
        .getInfo()
    )

    total_area = aoi.area(maxError=1).getInfo()
    permanent_area = float(permanent_area or 0.0)
    new_water_area = float(new_water_area or 0.0)

    return {
        "analysis_area_km2": round(total_area / 1_000_000, 6),
        "historical_permanent_water_km2": round(
            permanent_area / 1_000_000,
            6,
        ),
        "probable_new_water_km2": round(
            new_water_area / 1_000_000,
            6,
        ),
        "probable_new_water_percentage": round(
            new_water_area / max(total_area, 1.0) * 100.0,
            6,
        ),
    }


# ============================================================
# FOLIUM HELPERS
# ============================================================

def add_ee_layer(
    map_object: folium.Map,
    image,
    visualization: dict,
    name: str,
    shown: bool = True,
    opacity: float = 1.0,
) -> None:
    map_id = ee.Image(image).getMapId(visualization)

    folium.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=shown,
        opacity=opacity,
        max_zoom=20,
    ).add_to(map_object)


def create_map(pair: dict, layers: dict, summary: dict) -> folium.Map:
    map_object = folium.Map(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Esri, Maxar, Earthstar Geographics, and the GIS User Community"
        ),
        name="High-resolution satellite basemap",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(map_object)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(map_object)

    add_ee_layer(
        map_object,
        layers["latest_vh"],
        {
            "min": -28,
            "max": -5,
            "palette": ["000000", "666666", "FFFFFF"],
        },
        f"Latest Sentinel-1 VH — {pair['latest_time']} UTC",
        shown=False,
        opacity=0.80,
    )

    add_ee_layer(
        map_object,
        layers["vh_change"],
        {
            "min": -8,
            "max": 8,
            "palette": ["0033FF", "FFFFFF", "FF3300"],
        },
        "Sentinel-1 VH change (latest minus previous)",
        shown=False,
        opacity=0.80,
    )

    add_ee_layer(
        map_object,
        layers["permanent_water"],
        {
            "min": 0,
            "max": 1,
            "palette": ["00FFFF"],
        },
        "Historical permanent water",
        shown=True,
        opacity=0.70,
    )

    add_ee_layer(
        map_object,
        layers["probable_new_water"],
        {
            "min": 0,
            "max": 1,
            "palette": ["004CFF"],
        },
        "Probable new surface water",
        shown=True,
        opacity=0.90,
    )

    folium.Circle(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        radius=ANALYSIS_RADIUS_METERS,
        color="#ff3333",
        weight=2,
        fill=True,
        fill_color="#ff3333",
        fill_opacity=0.06,
        tooltip=f"{ANALYSIS_RADIUS_METERS} m analysis buffer",
    ).add_to(map_object)

    popup_html = (
        f"<b>{TARGET_NAME}</b><br>"
        f"Coordinates: {TARGET_LATITUDE}, {TARGET_LONGITUDE}<br>"
        f"Latest Sentinel-1: {pair['latest_time']} UTC<br>"
        f"Previous Sentinel-1: {pair['previous_time']} UTC<br>"
        f"Probable new water: {summary['probable_new_water_km2']} km²<br>"
        f"Buffer coverage: {summary['probable_new_water_percentage']}%"
    )

    folium.Marker(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        tooltip=TARGET_NAME,
        popup=folium.Popup(popup_html, max_width=350),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(map_object)

    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                z-index:9999;background:rgba(255,255,255,0.94);
                padding:10px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.28);
                font-family:Arial;text-align:center;">
      <b>Thaton Point — Sentinel-1 Water Detection</b><br>
      <span style="font-size:12px;">
        Latest: {pair['latest_time']} UTC |
        Reference: {pair['previous_time']} UTC
      </span>
    </div>
    """
    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(255,255,255,0.94);padding:10px 12px;
                border-radius:7px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:Arial;font-size:12px;">
      <b>Water layers</b><br>
      <span style="display:inline-block;width:13px;height:13px;background:#00FFFF;"></span>
      Historical permanent water<br>
      <span style="display:inline-block;width:13px;height:13px;background:#004CFF;"></span>
      Probable new surface water<br>
      <span style="display:inline-block;width:13px;height:13px;background:#FF3333;"></span>
      Target point / 1 km buffer<br>
      <i>Prototype thresholds; not an official flood map.</i>
    </div>
    """
    map_object.get_root().html.add_child(
        folium.Element(legend_html)
    )

    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    initialize_earth_engine()
    target, analysis_area = create_target_geometries()

    print()
    print("Finding latest same-orbit Sentinel-1 observations...")
    pair = get_latest_same_orbit_pair(analysis_area)

    print("Latest Sentinel-1:", pair["latest_time"], "UTC")
    print("Previous Sentinel-1:", pair["previous_time"], "UTC")
    print("Platform:", pair["platform"])
    print("Orbit pass:", pair["orbit_pass"])
    print("Relative orbit:", pair["relative_orbit"])

    print()
    print("Building permanent-water and probable-new-water layers...")
    layers = build_water_layers(analysis_area, pair)
    summary = calculate_area_summary(analysis_area, layers)

    result = {
        "target": {
            "name": TARGET_NAME,
            "latitude": TARGET_LATITUDE,
            "longitude": TARGET_LONGITUDE,
            "analysis_radius_m": ANALYSIS_RADIUS_METERS,
        },
        "latest_sentinel1_time_utc": pair["latest_time"],
        "previous_sentinel1_time_utc": pair["previous_time"],
        "platform": pair["platform"],
        "orbit_pass": pair["orbit_pass"],
        "relative_orbit": pair["relative_orbit"],
        "summary": summary,
        "thresholds": {
            "maximum_event_vh_db": MAX_EVENT_VH_DB,
            "minimum_vh_decrease_db": MIN_VH_DECREASE_DB,
            "maximum_slope_degrees": MAX_SLOPE_DEGREES,
            "permanent_water_occurrence_percent": (
                PERMANENT_WATER_OCCURRENCE
            ),
        },
        "method": (
            "Latest versus previous same-orbit Sentinel-1 VH change, "
            "low-slope filtering, and historical permanent-water removal."
        ),
        "warning": (
            "Research prototype only. Not an official flood map or warning."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    map_object = create_map(pair, layers, summary)
    map_object.save(str(OUTPUT_HTML))

    print()
    print("=" * 72)
    print("SENTINEL-1 WATER MAP CREATED")
    print("=" * 72)
    print("Probable new water:", summary["probable_new_water_km2"], "km²")
    print(
        "Buffer coverage:",
        summary["probable_new_water_percentage"],
        "%",
    )
    print("Map:", OUTPUT_HTML)
    print("Summary JSON:", OUTPUT_JSON)
    print("Internet is required when the HTML map is opened.")

   # webbrowser.open(OUTPUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()
