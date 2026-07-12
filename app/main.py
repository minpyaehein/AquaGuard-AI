from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .settings import settings
from .services.regional_demo import regional_snapshot, detail, COUNTRIES
from .services.thaton_case import snapshot as thaton_snapshot
from .services.evidence_store import EvidenceStore
STATIC=Path(__file__).resolve().parent/"static"
app=FastAPI(title=settings.app_name,version="3.0.0")
app.mount("/static",StaticFiles(directory=STATIC),name="static")
class Review(BaseModel):
    decision:str=Field(pattern="^(approve|reject)$");reviewer:str=Field(min_length=2);reason:str=Field(min_length=3);token:str
@app.get("/",include_in_schema=False)
def home():return FileResponse(STATIC/"index.html")
@app.get("/api/v3/health")
def health():return {"status":"ok","mode":"synthetic demonstration" if settings.demo_mode else "configured data"}
@app.get("/api/v3/region")
def region():return regional_snapshot()
@app.get("/api/v3/countries/{code}")
def country(code:str):
    code=code.upper()
    if code not in COUNTRIES:raise HTTPException(404,"Unknown ASEAN country code")
    return detail(code)

@app.get("/api/v3/case-studies/thaton")
def thaton_case():
    return thaton_snapshot()

@app.get("/api/v4/case-studies/thaton/evidence")
def thaton_evidence():
    ledger=Path("reports/thaton/evidence_ledger.json")
    if not ledger.exists():
        return {"status":"not_validated","message":"Add a real event manifest and run scripts/validate_thaton_event.py"}
    import json
    return json.loads(ledger.read_text(encoding="utf-8"))

@app.get("/api/v4/case-studies/thaton/metrics")
def thaton_metrics():
    report=Path("reports/thaton/segmentation_metrics.json")
    if not report.exists():
        return {"status":"not_evaluated","message":"Run training.evaluate_segmentation with real reference and prediction arrays"}
    import json
    return json.loads(report.read_text(encoding="utf-8"))

@app.post("/api/v3/alerts/{code}/review")
def review(code:str,req:Review):
    if req.token!=settings.authorized_reviewer_token:raise HTTPException(403,"Invalid reviewer token")
    if code.upper() not in COUNTRIES:raise HTTPException(404,"Unknown ASEAN country code")
    return {"country":code.upper(),"status":"approved_test" if req.decision=="approve" else "rejected","reviewer":req.reviewer,"reason":req.reason,"delivery":{"dashboard":"recorded","sms":"simulated","cell_broadcast":"authority integration required"}}
