from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,pandas as pd,yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from .common import seed_everything,binary_metrics,save_json
from app.ai.registry import ModelRegistry

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--csv",required=True);ap.add_argument("--out",required=True);ap.add_argument("--features",default="configs/features.yaml");ap.add_argument("--seed",type=int,default=42);args=ap.parse_args();seed_everything(args.seed)
    cfg=yaml.safe_load(Path(args.features).read_text());df=pd.read_csv(args.csv);num=cfg["risk_features"];cat=cfg["categorical_features"];label=cfg["label"]
    required=set(num+cat+[label,"event_id"]);missing=required-set(df.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed);train_idx,val_idx=next(splitter.split(df,groups=df.event_id));tr,va=df.iloc[train_idx],df.iloc[val_idx]
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
    model=HistGradientBoostingClassifier(max_depth=6,learning_rate=.06,max_iter=300,l2_regularization=1.0,random_state=args.seed)
    pipe=Pipeline([("prep",prep),("model",model)]);pipe.fit(tr[num+cat],tr[label]);prob=pipe.predict_proba(va[num+cat])[:,1];metrics=binary_metrics(va[label].to_numpy(),prob)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);bundle={"pipeline":pipe,"metadata":{"name":"asean-flood-risk","version":out.name,"features":num+cat,"validation_groups":"event_id"}};joblib.dump(bundle,out/"model.joblib")
    reports={"validation":metrics,"train_rows":len(tr),"validation_rows":len(va),"countries":sorted(df.country_code.astype(str).unique().tolist()),"events":int(df.event_id.nunique())};save_json(out/"metrics.json",reports)
    manifest={"name":"asean-flood-risk","version":out.name,"task":"risk_classification","validation_status":"candidate","metrics":metrics,"data_statement":"User-provided dataset; inspect provenance and licensing."};ModelRegistry(out.parent).register(out,manifest)
    print(json.dumps(reports,indent=2))
if __name__=="__main__":main()
