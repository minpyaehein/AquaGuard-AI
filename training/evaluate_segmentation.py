from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .common import save_json

def metrics(reference,prediction,threshold=.5,ignore=255):
    valid=reference!=ignore;y=(reference[valid]>0).astype(np.uint8);p=np.asarray(prediction[valid],dtype=float);hat=(p>=threshold).astype(np.uint8)
    tp=int(((hat==1)&(y==1)).sum());tn=int(((hat==0)&(y==0)).sum());fp=int(((hat==1)&(y==0)).sum());fn=int(((hat==0)&(y==1)).sum())
    div=lambda a,b:float(a/b) if b else 0.0
    return {"threshold":threshold,"valid_pixels":int(valid.sum()),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"iou_flood":div(tp,tp+fp+fn),"f1_flood":div(2*tp,2*tp+fp+fn),"precision_flood":div(tp,tp+fp),"recall_flood":div(tp,tp+fn),"specificity":div(tn,tn+fp),"false_alarm_rate":div(fp,fp+tn),"miss_rate":div(fn,fn+tp)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--reference",required=True,help=".npy reference mask: 0,1,255") ;ap.add_argument("--prediction",required=True,help=".npy flood probability [0,1]");ap.add_argument("--out",default="reports/thaton/segmentation_metrics.json");ap.add_argument("--threshold",type=float,default=.5);a=ap.parse_args()
    ref=np.load(a.reference);pred=np.load(a.prediction)
    if ref.shape!=pred.shape:raise ValueError(f"shape mismatch {ref.shape} != {pred.shape}")
    result=metrics(ref,pred,a.threshold);save_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
