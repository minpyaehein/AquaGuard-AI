from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import folium
import numpy as np
from folium.raster_layers import ImageOverlay
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "himawari_latest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "latest_thaton_cloud_basemap.html"


# ============================================================
# MAP SETTINGS
# ============================================================

# Same hard-coded target used by the point flood detector.
TARGET_NAME = "Thaton Target Point"
TARGET_LATITUDE = 16.91867
TARGET_LONGITUDE = 97.37001
ANALYSIS_RADIUS_METERS = 1000

# Area used for the Himawari overlay.
WEST = 96.90
SOUTH = 16.40
EAST = 97.85
NORTH = 17.40

BAND = "B13"

# B13 brightness-temperature display limits.
# Colder cloud tops become brighter and more opaque.
MIN_TEMPERATURE_K = 230.0
MAX_TEMPERATURE_K = 290.0
MIN_CLOUD_ALPHA = 0
MAX_CLOUD_ALPHA = 185


# ============================================================
# LOAD LATEST HIMAWARI FRAME
# ============================================================

def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/himawari_aws.py first."
        )

    return json.loads(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )


def validate_frame_files(frame: dict) -> list[str]:
    files = [Path(name) for name in frame.get("files", [])]

    if len(files) != 10:
        raise ValueError(
            f"Expected 10 Himawari segment files, found {len(files)}."
        )

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Himawari files:\n" + "\n".join(missing)
        )

    return [str(path) for path in files]


def load_latest_b13() -> tuple[np.ndarray, str]:
    manifest = load_manifest()
    frames = manifest.get("frames", [])

    if not frames:
        raise ValueError("No frames were found in the manifest.")

    latest_frame = frames[-1]
    timestamp = latest_frame["timestamp_utc"]
    files = validate_frame_files(latest_frame)

    print("Loading latest Himawari frame:", timestamp, "UTC")

    scene = Scene(
        filenames=files,
        reader="ahi_hsd",
    )
    scene.load([BAND])

    cropped = scene.crop(
        ll_bbox=(WEST, SOUTH, EAST, NORTH)
    )
    data = np.asarray(
        cropped[BAND].values,
        dtype=np.float32,
    )
    data[~np.isfinite(data)] = np.nan

    if not np.any(np.isfinite(data)):
        raise ValueError("No valid B13 pixels were found.")

    return data, timestamp


# ============================================================
# CREATE TRANSPARENT CLOUD OVERLAY
# ============================================================

def make_cloud_rgba(data: np.ndarray) -> np.ndarray:
    # Normalize temperature: 0 = warm, 1 = cold.
    clipped = np.clip(
        data,
        MIN_TEMPERATURE_K,
        MAX_TEMPERATURE_K,
    )
    coldness = (
        MAX_TEMPERATURE_K - clipped
    ) / (
        MAX_TEMPERATURE_K - MIN_TEMPERATURE_K
    )
    coldness = np.nan_to_num(coldness, nan=0.0)

    # White cloud overlay. Colder pixels are brighter.
    intensity = (
        110 + coldness * 145
    ).astype(np.uint8)

    # Warm pixels remain mostly transparent; cold cloud tops are clearer.
    alpha = np.clip(
        (coldness ** 1.35) * MAX_CLOUD_ALPHA,
        MIN_CLOUD_ALPHA,
        MAX_CLOUD_ALPHA,
    ).astype(np.uint8)

    rgba = np.zeros(
        (data.shape[0], data.shape[1], 4),
        dtype=np.uint8,
    )
    rgba[..., 0] = intensity
    rgba[..., 1] = intensity
    rgba[..., 2] = intensity
    rgba[..., 3] = alpha

    # Satpy array rows run north to south, matching Leaflet image bounds.
    return rgba


# ============================================================
# CREATE INTERACTIVE MAP
# ============================================================

def create_map(
    cloud_rgba: np.ndarray,
    timestamp: str,
) -> folium.Map:
    map_object = folium.Map(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        zoom_start=13,
        tiles=None,
        control_scale=True,
    )

    # High-resolution geographic context. These tiles load from the
    # internet when the HTML file is opened.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
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

    ImageOverlay(
        image=cloud_rgba,
        bounds=[[SOUTH, WEST], [NORTH, EAST]],
        opacity=0.72,
        name=f"Himawari-9 B13 clouds — {timestamp} UTC",
        interactive=True,
        cross_origin=False,
        zindex=5,
    ).add_to(map_object)

    folium.Circle(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        radius=ANALYSIS_RADIUS_METERS,
        color="#ff3333",
        weight=2,
        fill=True,
        fill_color="#ff3333",
        fill_opacity=0.08,
        tooltip=f"{ANALYSIS_RADIUS_METERS} m analysis buffer",
    ).add_to(map_object)

    folium.Marker(
        location=[TARGET_LATITUDE, TARGET_LONGITUDE],
        tooltip=TARGET_NAME,
        popup=(
            f"<b>{TARGET_NAME}</b><br>"
            f"Latitude: {TARGET_LATITUDE}<br>"
            f"Longitude: {TARGET_LONGITUDE}<br>"
            f"Himawari time: {timestamp} UTC"
        ),
        icon=folium.Icon(
            color="red",
            icon="info-sign",
        ),
    ).add_to(map_object)

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: rgba(255,255,255,0.92);
                padding: 10px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25);
                font-family: Arial; text-align: center;">
      <b>Thaton — High-resolution Basemap + Himawari-9 B13 Cloud Overlay</b><br>
      <span style="font-size: 12px;">Latest frame: {timestamp} UTC</span>
    </div>
    """
    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: rgba(255,255,255,0.92); padding: 10px 12px;
                border-radius: 7px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
                font-family: Arial; font-size: 12px;">
      <b>Layers</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:white;
                   border:1px solid #777;opacity:0.8;"></span>
      Colder/high cloud tops<br>
      <span style="display:inline-block;width:12px;height:12px;background:#ff3333;"></span>
      Target point / 1 km buffer<br>
      <i>Cloud overlay is approximately 2 km resolution.</i>
    </div>
    """
    map_object.get_root().html.add_child(
        folium.Element(legend_html)
    )

    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


def main():
    data, timestamp = load_latest_b13()
    cloud_rgba = make_cloud_rgba(data)
    map_object = create_map(cloud_rgba, timestamp)
    map_object.save(str(OUTPUT_HTML))

    print()
    print("Interactive map created successfully.")
    print("Output:", OUTPUT_HTML)
    print("The basemap requires an internet connection when opened.")

    # webbrowser.open(OUTPUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()
