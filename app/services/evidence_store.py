from __future__ import annotations
from pathlib import Path
import hashlib, json, yaml

class EvidenceStore:
    def __init__(self, root="data/case_studies/thaton"):
        self.root=Path(root)
    @staticmethod
    def checksum(path:Path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
        return h.hexdigest()
    def validate_manifest(self, manifest_path):
        p=Path(manifest_path);m=yaml.safe_load(p.read_text(encoding="utf-8"));errors=[];files=[]
        required=["event_id","event_start","event_end","sources","model"]
        for k in required:
            if not m.get(k):errors.append(f"missing manifest field: {k}")
        for name,src in (m.get("sources") or {}).items():
            q=Path(src.get("path", ""))
            exists=q.is_file()
            if not exists:errors.append(f"missing source file: {name} -> {q}")
            files.append({"name":name,"path":str(q),"exists":exists,"sha256":self.checksum(q) if exists else None,"scene_or_product_id":src.get("scene_id") or src.get("product_id"),"acquired_or_observed_at":src.get("acquired_at") or src.get("observed_at"),"license":src.get("license")})
        return {"valid":not errors,"errors":errors,"event_id":m.get("event_id"),"files":files,"manifest":m}
    def save_ledger(self, validation, out="reports/thaton/evidence_ledger.json"):
        p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(validation,indent=2),encoding="utf-8");return p
