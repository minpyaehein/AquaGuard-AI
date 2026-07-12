from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .unet import UNet
from .seg_dataset import NPZFloodDataset
from .common import seed_everything,save_json
from app.ai.registry import ModelRegistry

def scores(logits,y,valid):
    p=(torch.sigmoid(logits)>=.5)&valid.bool();t=(y>=.5)&valid.bool();inter=(p&t).sum().item();union=(p|t).sum().item();tp=inter;fp=(p&~t).sum().item();fn=(~p&t&valid.bool()).sum().item();return inter,union,tp,fp,fn

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--channels",type=int,required=True);ap.add_argument("--epochs",type=int,default=30);ap.add_argument("--batch-size",type=int,default=4);ap.add_argument("--lr",type=float,default=1e-3);ap.add_argument("--seed",type=int,default=42);args=ap.parse_args();seed_everything(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu";train=NPZFloodDataset(Path(args.data)/"train");val=NPZFloodDataset(Path(args.data)/"val")
    if not len(train) or not len(val):raise ValueError("Add .npz chips to train and val folders")
    model=UNet(args.channels).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none");best=-1;out=Path(args.out);out.mkdir(parents=True,exist_ok=True);history=[]
    for epoch in range(1,args.epochs+1):
        model.train();running=0
        for x,y,v in DataLoader(train,args.batch_size,shuffle=True):
            x,y,v=x.to(device),y.to(device),v.to(device);opt.zero_grad();raw=loss_fn(model(x),y);loss=(raw*v).sum()/v.sum().clamp_min(1);loss.backward();opt.step();running+=loss.item()
        model.eval();tot=[0,0,0,0,0]
        with torch.no_grad():
            for x,y,v in DataLoader(val,args.batch_size):
                q=scores(model(x.to(device)),y.to(device),v.to(device));tot=[a+b for a,b in zip(tot,q)]
        iou=tot[0]/max(tot[1],1);f1=2*tot[2]/max(2*tot[2]+tot[3]+tot[4],1);row={"epoch":epoch,"loss":running/max(len(train),1),"val_iou":iou,"val_f1":f1};history.append(row);print(row)
        if iou>best:best=iou;torch.save({"state_dict":model.state_dict(),"in_channels":args.channels,"architecture":"UNet"},out/"model.pt")
    save_json(out/"metrics.json",{"best_val_iou":best,"history":history});ModelRegistry(out.parent).register(out,{"name":"asean-flood-segmentation","version":out.name,"task":"flood_segmentation","validation_status":"candidate","metrics":{"best_val_iou":best},"input_contract":"float32 [B,C,H,W]"})
if __name__=="__main__":main()
