from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET_NAME = "noaa-himawari9"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "himawari"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_FILE = PROCESSED_DIR / "latest_himawari_download.json"

DEFAULT_BAND = "B13"
DEFAULT_FRAME_COUNT = 7
FRAME_INTERVAL_MINUTES = 10

# New Himawari observations can appear after the nominal observation time.
# The search walks backwards until it finds complete timestamps.
SEARCH_BACK_MINUTES = 360
EXPECTED_SEGMENTS = 10


def create_s3_client():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )


def floor_to_ten_minutes(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    return value.replace(
        minute=(value.minute // 10) * 10,
        second=0,
        microsecond=0,
    )


def timestamp_prefix(timestamp: datetime) -> str:
    return timestamp.strftime("AHI-L1b-FLDK/%Y/%m/%d/%H%M/")


def destination_directory(timestamp: datetime, band: str) -> Path:
    return (
        RAW_ROOT
        / timestamp.strftime("%Y-%m-%d")
        / timestamp.strftime("%H%M")
        / band
    )


def list_all_keys(s3_client, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=prefix,
    ):
        for item in page.get("Contents", []):
            keys.append(item["Key"])

    return keys


def select_band_keys(keys: list[str], band: str) -> list[str]:
    marker = f"_{band}_"
    return sorted(
        key for key in keys
        if marker in Path(key).name
    )


def find_latest_complete_timestamps(
    s3_client,
    band: str,
    frame_count: int,
) -> list[dict]:
    search_time = floor_to_ten_minutes(
        datetime.now(timezone.utc)
    )

    complete: list[dict] = []
    checked = 0
    maximum_checks = SEARCH_BACK_MINUTES // FRAME_INTERVAL_MINUTES

    print("Searching NOAA AWS for complete Himawari observations...")
    print("Starting from UTC:", search_time.isoformat())

    while checked <= maximum_checks and len(complete) < frame_count:
        prefix = timestamp_prefix(search_time)
        all_keys = list_all_keys(s3_client, prefix)
        band_keys = select_band_keys(all_keys, band)

        print(
            search_time.strftime("%Y-%m-%d %H:%M UTC"),
            "->",
            len(band_keys),
            f"{band} segments",
        )

        if len(band_keys) == EXPECTED_SEGMENTS:
            complete.append({
                "timestamp": search_time,
                "prefix": prefix,
                "keys": band_keys,
            })

        search_time -= timedelta(minutes=FRAME_INTERVAL_MINUTES)
        checked += 1

    if len(complete) < frame_count:
        raise RuntimeError(
            f"Only {len(complete)} complete {band} timestamps were found "
            f"within the last {SEARCH_BACK_MINUTES} minutes; "
            f"{frame_count} were requested."
        )

    # Return oldest to newest for temporal processing.
    return list(reversed(complete))


def download_files(
    s3_client,
    keys: list[str],
    destination_dir: Path,
) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for key in keys:
        destination = destination_dir / Path(key).name

        if destination.exists() and destination.stat().st_size > 0:
            print("Already exists:", destination.name)
        else:
            print("Downloading:", destination.name)
            s3_client.download_file(
                BUCKET_NAME,
                key,
                str(destination),
            )

        downloaded.append(destination)

    return downloaded


def save_manifest(records: list[dict], band: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "NOAA AWS Himawari-9 AHI-L1b-FLDK",
        "bucket": BUCKET_NAME,
        "band": band,
        "frame_interval_minutes": FRAME_INTERVAL_MINUTES,
        "frame_count": len(records),
        "oldest_time_utc": records[0]["timestamp_utc"],
        "latest_time_utc": records[-1]["timestamp_utc"],
        "frames": records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    MANIFEST_FILE.write_text(
        json.dumps(payload, indent=4),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest complete Himawari-9 Full Disk "
            "observations at 10-minute intervals."
        )
    )
    parser.add_argument(
        "--band",
        default=DEFAULT_BAND,
        help="AHI band, for example B13.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help="Number of complete 10-minute observations to download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.frames < 1:
        raise ValueError("--frames must be at least 1.")

    band = args.band.upper()
    s3_client = create_s3_client()

    print("Connecting to:", BUCKET_NAME)
    print("Selected band:", band)
    print("Requested frames:", args.frames)
    print()

    observations = find_latest_complete_timestamps(
        s3_client=s3_client,
        band=band,
        frame_count=args.frames,
    )

    manifest_records: list[dict] = []

    for index, observation in enumerate(observations, start=1):
        timestamp = observation["timestamp"]
        destination_dir = destination_directory(timestamp, band)

        print()
        print("=" * 70)
        print(
            f"FRAME {index}/{len(observations)}:",
            timestamp.strftime("%Y-%m-%d %H:%M UTC"),
        )
        print("=" * 70)

        files = download_files(
            s3_client=s3_client,
            keys=observation["keys"],
            destination_dir=destination_dir,
        )

        manifest_records.append({
            "timestamp_utc": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H%M"),
            "prefix": observation["prefix"],
            "segment_count": len(files),
            "download_directory": str(destination_dir),
            "files": [str(path) for path in files],
        })

    save_manifest(manifest_records, band)

    print()
    print("=" * 70)
    print("LATEST HIMAWARI DOWNLOAD COMPLETED")
    print("=" * 70)
    print("Oldest frame:", manifest_records[0]["timestamp_utc"], "UTC")
    print("Latest frame:", manifest_records[-1]["timestamp_utc"], "UTC")
    print("Frames:", len(manifest_records))
    print("Segments per frame:", EXPECTED_SEGMENTS)
    print("Manifest:", MANIFEST_FILE)


if __name__ == "__main__":
    main()
