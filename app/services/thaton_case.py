from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json, math, yaml
CFG=yaml.safe_load(Path("configs/thaton.yaml").read_text(encoding="utf-8"))

def ring(cx,cy,dx,dy):return [[cx-dx,cy-dy],[cx+dx,cy-dy*.8],[cx+dx*.75,cy+dy],[cx-dx*.8,cy+dy*.75],[cx-dx,cy-dy]]
def feature(name,kind,prob,coords,confidence,source_time):
 return {"type":"Feature","geometry":{"type":"Polygon","coordinates":[coords]},"properties":{"name":name,"kind":kind,"probability":prob,"confidence":confidence,"source_time":source_time,"status":"synthetic historical-replay demo"}}
def snapshot():
 now=datetime.now(timezone.utc);cx,cy=97.3714,16.9206
 risk=[feature("Thaton West","predicted_risk",.86,ring(cx-.022,cy+.003,.016,.012),.81,(now-timedelta(minutes=10)).isoformat()),feature("Thaton Central","predicted_risk",.72,ring(cx+.006,cy,.014,.010),.78,(now-timedelta(minutes=10)).isoformat()),feature("Southern approach","predicted_risk",.58,ring(cx+.004,cy-.027,.018,.011),.70,(now-timedelta(minutes=10)).isoformat())]
 water=[feature("Model-observed water A","observed_inundation",.82,ring(cx-.026,cy+.001,.010,.007),.84,(now-timedelta(hours=18)).isoformat()),feature("Model-observed water B","observed_inundation",.69,ring(cx+.006,cy-.018,.009,.006),.73,(now-timedelta(hours=18)).isoformat())]
 timeline=[]
 for i in range(18):
  r=max(.12,min(.94,.18+i*.042+.08*math.sin(i/2)));timeline.append({"step":i,"label":f"T-{(17-i)*10}m" if i<17 else "Now","risk":round(r,3),"rainfall_demo":round(5+82*r,1),"river_demo":round(.18+.7*r,2)})
 return {"case":CFG,"generated_at":now.isoformat(),"mode":"synthetic historical replay","metrics":{"flood_probability":.86,"confidence":.81,"impact_priority":.63,"pending_review":True},"map":{"risk":{"type":"FeatureCollection","features":risk},"inundation":{"type":"FeatureCollection","features":water}},"timeline":timeline,"evidence":[{"name":"Himawari-9 weather features","updated":"10 min","quality":.84,"role":"rapid risk"},{"name":"GPM / rainfall inputs","updated":"30 min","quality":.78,"role":"rainfall evidence"},{"name":"River gauge adapter","updated":"5 min","quality":.72,"role":"water-level trend"},{"name":"Sentinel-1 SAR","updated":"18 h","quality":.88,"role":"inundation evidence"},{"name":"Terrain and river proximity","updated":"baseline","quality":.90,"role":"susceptibility"}],"impact":{"population_demo":18450,"roads_km_demo":28.6,"health_sites_demo":3,"shelters_demo":5,"telecom_sites_demo":7,"network_availability_demo":.71},"explain":{"drivers":[{"name":"6-hour rainfall","contribution":.31},{"name":"river level trend","contribution":.24},{"name":"terrain susceptibility","contribution":.19},{"name":"SAR water evidence","contribution":.18},{"name":"other features","contribution":.08}],"limitations":["Values on this screen are synthetic for competition demonstration.","Risk probability is not the same as confirmed inundation.","Public warning requires authorized review and real agency/operator integration."]}}
