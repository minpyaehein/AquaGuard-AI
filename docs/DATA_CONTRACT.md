# Data contract

## Risk observations CSV
Required columns are defined in `configs/features.yaml`. `flooded` must be 0 or 1. `event_id` must identify an independent flood event; splitting random pixels from the same event across train and validation causes leakage. `country_code` uses two-letter codes in `configs/countries.yaml`.

Recommended provenance columns: `source`, `license`, `processing_version`, `quality_flag`, `label_method`, `reviewer`, and `geometry_id`.

## Segmentation chips
NPZ keys: `image [C,H,W] float32`, `mask [H,W] uint8`; mask values 0 background, 1 flood, 255 ignore. Split by event and geography, not random neighboring tiles.

## Model acceptance gate
A model is only a candidate until it has: country-held-out evaluation; event-held-out evaluation; calibration review; false-alarm analysis; cloud/no-data handling; model card; and authorized approval. Set `validation_status` to `approved` only after the gate is documented.
