"""
This script creates the labels for euronews dataset. 
"""

import os
from pathlib import Path
import json
import torch
from tqdm import tqdm
import math

embedding_root = "/data/euronews_dataset/processed_mono_embeddings/"
labels_root = "/data/euronews_dataset/euronews/"

embedding_files = list(Path(embedding_root).glob("**/*.pt"))
prog = tqdm(embedding_files)

for file in prog:
    try:
        emb = torch.load(file)
        if isinstance(emb, dict):
            emb = emb['embeddings']
        
        filename = file.stem
        labels_path = os.path.join(labels_root, file.parent.stem, f"{filename}.json")
        label_file = json.load(open(labels_path))
        
        labels = torch.zeros(emb.shape[0])
        labels[-1]=1
        chapters = [math.ceil(i['start_time'] / 10) for i in label_file['chapters'] if i['start_time']!=0]
        if chapters[-1]==emb.shape[0]:
            chapters = chapters[:-1]
        labels[chapters]=1
        output = {
            "embeddings": emb,
            "labels": labels,
            "chapters": label_file['chapters'],
            "duration": label_file['duration']
        }
        # breakpoint()
        torch.save(output, file)
    except Exception as e:
        print(file, e)
        # breakpoint()