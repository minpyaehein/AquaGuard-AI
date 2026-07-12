from __future__ import annotations
from datetime import datetime, timezone
import hashlib, math
from pathlib import Path
import yaml
COUNTRIES = yaml.safe_load(Path("configs/countries.yaml").read_text(encoding="utf-8"))["countries"]

def stable(code:str, salt:str)->float:
    n=int(hashlib.sha256(f"{code}:{salt}".encode()).hexdigest()[:8],16)
    return (n%1000)/1000

def country_card(code:str):
    c=COUNTRIES[code]; p=.18+.72*stable(code,"risk"); conf=.62+.32*stable(code,"confidence")
    level="critical" if p>=.82 else "high" if p>=.65 else "moderate" if p>=.4 else "low"
    return {"code":code,"name":c["name"],"iso3":c["iso3"],"languages":c["languages"],"center":c["center"],"probability":round(p,3),"confidence":round(conf,3),"level":level,"status":"demo"}

def regional_snapshot():
    cards=[country_card(c) for c in COUNTRIES]
    return {"generated_at":datetime.now(timezone.utc).isoformat(),"mode":"synthetic demonstration","countries":cards,"summary":{"countries_monitored":len(cards),"high_or_critical":sum(x["level"] in {"high","critical"} for x in cards),"active_sources":5,"pending_reviews":1}}

def detail(code:str):
    card=country_card(code); lat,lon=card["center"]; r=.6+.8*stable(code,"radius")
    ring=[[lon-r,lat-r*.55],[lon+r,lat-r*.55],[lon+r*.8,lat+r*.55],[lon-r*.8,lat+r*.55],[lon-r,lat-r*.55]]
    timeline=[]
    for i in range(12):
        q=max(0,min(1,card["probability"]-.22+(.44*i/11)+.05*math.sin(i)))
        timeline.append({"step":i,"risk":round(q,3),"rain":round(20+65*q,1)})
    return {**card,"polygon":{"type":"Feature","geometry":{"type":"Polygon","coordinates":[ring]},"properties":{"probability":card["probability"],"level":card["level"]}},"timeline":timeline,"sources":[{"name":"Himawari-9","quality":.84,"age":"10 min"},{"name":"GPM IMERG","quality":.78,"age":"30 min"},{"name":"River gauges","quality":.73,"age":"5 min"},{"name":"Sentinel-1","quality":.88,"age":"18 h"},{"name":"Sentinel-2","quality":.52,"age":"2 d"}],"impact":{"population_demo":int(7000+70000*stable(code,"pop")),"roads_km_demo":round(8+45*stable(code,"roads"),1),"health_sites_demo":int(1+8*stable(code,"health")),"network_availability_demo":round(.55+.4*stable(code,"network"),2)}}
