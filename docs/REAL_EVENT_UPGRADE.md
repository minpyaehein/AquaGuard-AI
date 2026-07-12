# Upgrade point: real Thaton evidence and measured AI performance

This V4 targets the largest competition weakness: replacing synthetic demonstrations with one traceable historical event.

## Required artifacts
1. A completed event manifest copied from `configs/thaton_event_manifest.example.yaml`.
2. Real pre-event and event Earth-observation files with scene IDs and acquisition times.
3. Real rainfall/gauge input with product ID and timestamp.
4. An independently reviewed reference flood mask.
5. A model probability raster/array created without using the reference labels as inference input.

## Evidence validation
```powershell
python scripts/validate_thaton_event.py --manifest configs/thaton_event_manifest.yaml
```
The command writes `reports/thaton/evidence_ledger.json`, including missing-file errors and SHA-256 checksums.

## Segmentation evaluation
Convert the reference mask to a NumPy array with values 0, 1 and optionally 255 for ignore. Convert model output to a same-shape probability array.
```powershell
python -m training.evaluate_segmentation --reference data/case_studies/thaton/reference/reference.npy --prediction data/case_studies/thaton/predictions/probability.npy --out reports/thaton/segmentation_metrics.json
```
The report includes Flood IoU, F1, precision, recall, specificity, false-alarm rate and miss rate.

## Leakage prevention
Group tiles by flood event and location. Do not place neighboring chips from one source image in both training and test sets. Do not derive the prediction from the reference mask.

## Competition evidence panel
The API exposes:
- `GET /api/v4/case-studies/thaton/evidence`
- `GET /api/v4/case-studies/thaton/metrics`
Until real reports exist, these endpoints explicitly return `not_validated` or `not_evaluated`.
