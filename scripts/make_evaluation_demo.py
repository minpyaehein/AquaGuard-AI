"""Creates synthetic masks solely to verify the evaluation pipeline."""
from pathlib import Path
import numpy as np
rng=np.random.default_rng(7);h=w=256;y,x=np.ogrid[:h,:w];ref=(((x-128)**2/(75**2)+(y-135)**2/(46**2))<1).astype(np.uint8);prob=np.clip(ref*.72+rng.normal(.18,.16,(h,w)),0,1);out=Path("data/case_studies/thaton/predictions");out.mkdir(parents=True,exist_ok=True);np.save(out/"reference_demo.npy",ref);np.save(out/"probability_demo.npy",prob);print("Synthetic evaluation inputs created; do not report as real performance.")
