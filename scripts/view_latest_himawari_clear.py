from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from satpy import Scene


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_himawari_download.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "himawari_latest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Display area around Thaton. This is only for viewing the satellite image.
WEST = 96.90
SOUTH = 16.40
EAST = 97.85
NORTH = 17.40

BAND = "B13"


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run himawari_aws.py first."
        )

    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def validate_files(file_names: list[str]) -> list[str]:
    files = [str(Path(name)) for name in file_names]
    missing = [name for name in files if not Path(name).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing Himawari segment files:\n" + "\n".join(missing)
        )

    if len(files) != 10:
        raise ValueError(
            f"Expected 10 segment files, but received {len(files)}."
        )

    return files


def create_frame(frame: dict) -> Path:
    timestamp = frame["timestamp_utc"]
    files = validate_files(frame["files"])

    print(f"Processing {timestamp} UTC ...")

    scene = Scene(
        filenames=files,
        reader="ahi_hsd",
    )
    scene.load([BAND])

    # Crop the full-disk observation to the Thaton surroundings.
    cropped = scene.crop(ll_bbox=(WEST, SOUTH, EAST, NORTH))
    data = np.asarray(cropped[BAND].values, dtype=np.float32)
    data[~np.isfinite(data)] = np.nan

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError(
            f"No valid {BAND} pixels were found for {timestamp}."
        )

    # Use one fixed temperature scale for every frame so the
    # 10-minute images can be compared directly.
    vmin = 230.0
    vmax = 290.0

    output_file = OUTPUT_DIR / (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace(" ", "_")
        + f"_thaton_{BAND}.png"
    )

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(
        data,
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=[WEST, EAST, SOUTH, NORTH],
        interpolation="bilinear",
    )

    axis.scatter(
        [97.37001],
        [16.91867],
        color="red",
        edgecolor="white",
        s=65,
        label="Hard-coded target point",
        zorder=3,
    )
    axis.set_title(
        "Himawari-9 B13 Infrared Satellite Image\n"
        f"Thaton Area | {timestamp} UTC",
        fontweight="bold",
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25, linestyle="--")

    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Brightness Temperature (K)")

    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("Created:", output_file)
    return output_file


def create_contact_sheet(image_files: list[Path], manifest: dict) -> Path:
    images = [plt.imread(path) for path in image_files]
    columns = 4
    rows = int(np.ceil(len(images) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4.2 * rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index])
            axis.set_title(
                manifest["frames"][index]["timestamp_utc"] + " UTC",
                fontsize=10,
            )

    figure.suptitle(
        "Latest Himawari-9 B13 Observations — 10-Minute Sequence",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = OUTPUT_DIR / "latest_thaton_himawari_sequence.png"
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("Sequence sheet:", output_file)
    return output_file


def main():
    manifest = load_manifest()
    frames = manifest.get("frames", [])

    if not frames:
        raise ValueError("The Himawari manifest contains no frames.")

    print("Frames in manifest:", len(frames))
    print("Band:", manifest.get("band"))
    print("Oldest:", manifest.get("oldest_time_utc"), "UTC")
    print("Latest:", manifest.get("latest_time_utc"), "UTC")
    print()

    image_files = [create_frame(frame) for frame in frames]
    contact_sheet = create_contact_sheet(image_files, manifest)

    print()
    print("=" * 70)
    print("HIMAWARI VIEWING OUTPUT COMPLETED")
    print("=" * 70)
    print("Individual images:", OUTPUT_DIR)
    print("10-minute sequence:", contact_sheet)


if __name__ == "__main__":
    main()
