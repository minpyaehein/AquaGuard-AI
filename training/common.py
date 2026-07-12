from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except ImportError: pass

def save_json(path:str|Path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def binary_metrics(y_true,y_prob,threshold=.5):
    from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,brier_score_loss
    y_pred=(np.asarray(y_prob)>=threshold).astype(int)
    return {"roc_auc":float(roc_auc_score(y_true,y_prob)),"pr_auc":float(average_precision_score(y_true,y_prob)),"f1":float(f1_score(y_true,y_pred,zero_division=0)),"precision":float(precision_score(y_true,y_pred,zero_division=0)),"recall":float(recall_score(y_true,y_pred,zero_division=0)),"brier":float(brier_score_loss(y_true,y_prob)),"threshold":float(threshold)}
