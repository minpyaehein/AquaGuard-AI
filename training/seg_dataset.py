from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
class NPZFloodDataset(Dataset):
    def __init__(self,root):self.files=sorted(Path(root).glob("*.npz"));
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        import torch
        z=np.load(self.files[i],allow_pickle=False);x=z["image"].astype("float32");y=z["mask"].astype("float32");valid=y!=255;y=np.where(valid,y,0)
        return torch.from_numpy(x),torch.from_numpy(y[None]),torch.from_numpy(valid[None])
