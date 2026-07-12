# Competition readiness checklist
- [ ] Replace synthetic Thaton replay with one licensed historical event dataset.
- [ ] Preserve source timestamps and processing versions.
- [ ] Validate risk labels independently from model inputs.
- [ ] Fine-tune Sentinel-1 segmentation using Myanmar/ASEAN samples.
- [ ] Report event-held-out and country-held-out metrics.
- [ ] Add at least one uncertainty/failure case.
- [ ] Use validated population, road, health, shelter and telecom layers.
- [ ] Add Myanmar and English alert copy reviewed by fluent humans.
- [ ] Keep CAP status TEST during competition.
- [ ] Record model version, reviewer and alert decision in audit log.
- [ ] Prepare offline screenshots/video in case venue internet is unavailable.

## V4 evidence gate
- [ ] Complete `configs/thaton_event_manifest.yaml` with real IDs, timestamps and licenses.
- [ ] Run evidence validation with zero missing-file errors.
- [ ] Save evidence ledger and artifact checksums.
- [ ] Evaluate a model prediction against an independent reference mask.
- [ ] Show IoU, F1, precision, recall, false-alarm rate and miss rate in the pitch.
