# AquaGuard AI Geo — ASEAN Regional Edition

Competition-oriented, safety-conscious code extension for a regional flood-risk and inundation intelligence platform covering all 11 ASEAN Member States.

## What this adds
- 11-country configuration and regional deployment profiles
- Pluggable AI model interface: swap your own risk or segmentation model without changing the API
- Tabular risk-model training, U-Net segmentation training, calibration, evaluation, and model-card generation
- Country-held-out evaluation to test geographic generalization
- Model registry with version, checksum, metrics, and activation state
- Explicit separation of risk prediction, observed inundation, confidence, and authorized alerts
- Multilingual alert-template placeholders

## Install
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-core.txt
pip install -r requirements-ml.txt
copy .env.example .env
```

## Prepare tabular risk data
Create `data/processed/risk_features.csv` with the schema in `docs/DATA_CONTRACT.md`.

## Train risk model
```powershell
python -m training.train_risk --csv data/processed/risk_features.csv --out models/registry/risk_v1
```

## Prepare segmentation data
Create `.npz` chips under `data/processed/segmentation/{train,val,test}`. Each file must contain:
- `image`: float32 `[C,H,W]`
- `mask`: uint8 `[H,W]`, classes 0=background, 1=flood, 255=ignore
- optional `country_code`, `event_id`, `timestamp`

## Train segmentation model
```powershell
python -m training.train_segmentation --data data/processed/segmentation --out models/registry/seg_unet_v1 --channels 2
```

## Evaluate geographic generalization
```powershell
python -m training.evaluate_cross_country --csv data/processed/risk_features.csv --out reports/cross_country.json
```

## Add your own model
1. Copy `app/ai/plugins/custom_template.py`.
2. Implement `load()` and `predict()`.
3. Set `RISK_MODEL_PLUGIN=app.ai.plugins.your_module:YourModel` in `.env`.
4. Put weights under `models/custom/`; weights are intentionally not included.

## Truthful competition claim
This code is a strong engineering foundation, not a guarantee of winning and not an operational public-warning authority. Demonstrate validated data, country-held-out results, uncertainty, auditability, local-language communication, and human authorization.

## Run the competition command center
```powershell
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`. The regional values shown by the included server are synthetic demonstration data until you wire real adapters and validated models.

## Flagship Thaton pilot
The home dashboard now includes a dedicated Thaton Flood Intelligence Lab. It is a synthetic historical replay until you connect one licensed event dataset. See `docs/evidence/THATON_CASE_RATIONALE.md`, `docs/THATON_DEMO_SCRIPT.md`, and `docs/COMPETITION_READINESS_CHECKLIST.md`.

## V4: real-evidence gate
This edition adds a Thaton historical-event manifest, evidence ledger/checksums, and segmentation evaluation. See `docs/REAL_EVENT_UPGRADE.md`. The API will not imply validation when reports are absent.
