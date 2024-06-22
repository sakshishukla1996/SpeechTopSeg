"""
This script creates the labels for euronews dataset. 
"""

import os
import re
from pathlib import Path
import json
import torch
from tqdm import tqdm
import math

embedding_root = "/data/euronews_dataset/processed_mono_embeddings/it/"
labels_root = "/data/euronews_dataset/euronews/it/"

embedding_files = list(Path(embedding_root).glob("*.pt"))
prog = tqdm(embedding_files)

def extract_date_from_url(url):
    # Regular expression to match date in the format YYYY/MM/DD
    date_pattern = r'\b\d{4}/\d{2}/\d{2}\b'

    # Search for date pattern in the URL string
    match = re.search(date_pattern, url)

    if match:
        return match.group()
    else:
        return None

for file in prog:
    try:
        emb = torch.load(file)
        if isinstance(emb, dict):
            emb = emb['embeddings']
        filename = file.stem
        labels_path = os.path.join(labels_root, file.parent.stem, f"{filename}.json")
        if not os.path.exists(os.path.join(labels_root, file.parent.stem)):
            Path(os.path.join(labels_root, file.parent.stem)).mkdir(parents=True, exist_ok=True)
        label_file = json.load(open(labels_path))
        
        labels = torch.zeros(emb.shape[0])
        labels[-1]=1
        chapters = [math.ceil(i['start_time'] / 10) for i in label_file['chapters'] if i['start_time']!=0]
        if chapters[-1]==emb.shape[0]:
            chapters = chapters[:-1]
        labels[chapters]=1
        date = extract_date_from_url(label_file['description'])
        if date=="" or date==None:
            date = label_file['upload_date']
            date = f"{date[:4]}/{date[4:6]}/{date[6:8]}"
        output = {
            "embeddings": emb,
            "labels": labels,
            "chapters": label_file['chapters'],
            "duration": label_file['duration'],
            "date": date,
            "date_format": "YYYY/MM/DD"
        }
        torch.save(output, file)
    except Exception as e:
        print(file, e)
        # breakpoint()