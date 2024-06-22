"""
This script creates the labels for euronews spanish dataset. 
"""

import os
import re
from pathlib import Path
import json
import torch
from tqdm import tqdm
import math

yt_root ="/data/euronews_dataset/euronews/es/"
# yt_files = list(Path(yt_root).glob("*.json"))

embedding_root = "/data/euronews_dataset/processed_mono_embeddings/es/"
embedding_files = list(Path(embedding_root).glob("*.pt"))
for file in embedding_files:
    f = torch.load(file)
    if (f['date']=="" or f['date']==None):
        data = json.load(open(os.path.join(yt_root,f"{file.stem}.json")))
        a = data['upload_date']
        date = f"{a[:4]}/{a[4:6]}/{a[6:8]}"
        f['date'] = date
        print(file, date)
        torch.save(f, file)