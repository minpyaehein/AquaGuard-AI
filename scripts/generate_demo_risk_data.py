"""Synthetic data generator for pipeline testing only; never report its metrics as real model performance."""
from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(42);countries=["BN","KH","ID","LA","MY","MM","PH","SG","TH","TL","VN"];rows=[]
for c in countries:
 for event in range(12):
  for i in range(80):
   rain=rng.gamma(2,20);river=np.clip(rng.normal(.45+.003*rain,.2),0,1);slope=rng.random();prox=rng.random();z=-3+.025*rain+1.7*river+1.1*prox-.7*slope+rng.normal(0,.6);f=int(rng.random()<1/(1+np.exp(-z)));rows.append({"country_code":c,"event_id":f"{c}-{event}","observed_at":"2026-01-01T00:00:00Z","latitude":0,"longitude":0,"cold_cloud":rng.random(),"cloud_growth":rng.random(),"moisture":rng.random(),"rain_30m":rain/6,"rain_1h":rain/3,"rain_3h":rain*.65,"rain_6h":rain,"rain_24h":rain*1.8,"river_level":river,"river_rise_rate":rng.random(),"elevation_norm":rng.random(),"slope_norm":slope,"hand_norm":rng.random(),"river_proximity":prox,"soil_wetness":rng.random(),"flooded":f})
out=Path("data/processed/risk_features_demo.csv");out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);print(f"Wrote {out}; SYNTHETIC DATA ONLY")
