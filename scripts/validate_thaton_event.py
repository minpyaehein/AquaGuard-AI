import argparse,json
from app.services.evidence_store import EvidenceStore
p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--out",default="reports/thaton/evidence_ledger.json");a=p.parse_args();s=EvidenceStore();r=s.validate_manifest(a.manifest);s.save_ledger(r,a.out);print(json.dumps(r,indent=2));raise SystemExit(0 if r["valid"] else 2)
