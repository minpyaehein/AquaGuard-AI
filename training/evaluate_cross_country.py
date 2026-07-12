from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd,yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .common import binary_metrics,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--csv",required=True);ap.add_argument("--out",required=True);ap.add_argument("--features",default="configs/features.yaml");args=ap.parse_args();cfg=yaml.safe_load(Path(args.features).read_text());df=pd.read_csv(args.csv);num,cat,label=cfg["risk_features"],cfg["categorical_features"],cfg["label"];results={}
    for country in sorted(df.country_code.unique()):
        tr=df[df.country_code!=country];te=df[df.country_code==country]
        if te[label].nunique()<2 or tr[label].nunique()<2:results[country]={"skipped":"both classes required"};continue
        prep=ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
        pipe=Pipeline([("prep",prep),("model",HistGradientBoostingClassifier(max_depth=6,max_iter=250,random_state=42))]);pipe.fit(tr[num+cat],tr[label]);results[country]={**binary_metrics(te[label].to_numpy(),pipe.predict_proba(te[num+cat])[:,1]),"rows":len(te)}
    save_json(args.out,{"method":"leave-one-country-out","results":results});print(results)
if __name__=="__main__":main()
