from pathlib import Path
import joblib, numpy as np
from ..contracts import ModelPlugin, ModelOutput, PredictionContext

class SklearnRiskModel(ModelPlugin):
    task = "risk_classification"
    def __init__(self): self.pipeline=None; self.meta={}; self.path=None
    def load(self, path: Path, device: str="cpu"):
        self.path=path; bundle=joblib.load(path); self.pipeline=bundle["pipeline"]; self.meta=bundle.get("metadata",{})
    def predict(self, inputs, context: PredictionContext):
        p=np.asarray(self.pipeline.predict_proba(inputs)[:,1], dtype=np.float32)
        # confidence is calibrated-distance from 0.5; production systems should use validation-based calibration
        c=np.clip(np.abs(p-.5)*2,0,1).astype(np.float32)
        return ModelOutput(p,c,self.meta.get("name","sklearn-risk"),self.meta.get("version","unknown"),{"country":context.country_code})
    def health(self): return {"loaded":self.pipeline is not None,"task":self.task,"path":str(self.path)}
