"""Copy this file and replace the placeholder with your own model.
Expected segmentation input: float32 NumPy array [B,C,H,W].
Expected output: flood probability [B,H,W] in [0,1].
"""
from pathlib import Path
import numpy as np
from ..contracts import ModelPlugin, ModelOutput, PredictionContext

class CustomSegmentationModel(ModelPlugin):
    task="flood_segmentation"
    def __init__(self): self.model=None; self.path=None; self.device="cpu"
    def load(self, path:Path, device:str="cpu"):
        self.path=path; self.device=device
        if not path.exists():
            raise FileNotFoundError(f"Put your model at {path}, or configure another plugin")
        # Example for PyTorch:
        # import torch
        # self.model = torch.jit.load(str(path), map_location=device).eval()
        raise NotImplementedError("Implement model loading for your architecture")
    def predict(self, inputs, context:PredictionContext):
        # Example:
        # with torch.no_grad(): logits=self.model(torch.from_numpy(inputs).to(self.device))
        # p=torch.sigmoid(logits).squeeze(1).cpu().numpy()
        raise NotImplementedError("Implement inference and return ModelOutput")
    def health(self): return {"loaded":self.model is not None,"task":self.task,"path":str(self.path)}
