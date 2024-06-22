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
from src.utils import get_seg_boundaries, pk, win_diff


# embedding_root = "/data/euronews_dataset/processed_mono_embeddings/it/"

prediction_root = "/data/akashvani/all_all_all_predictions/"
labels_root = "/data/akashvani/prob_labels/"
output_root = "/data/akashvani/results/all_all_all/"

prediction_files = list(Path(prediction_root).glob("*.pt"))
prog = tqdm(prediction_files)

def extract_date_from_url(url):
    # Regular expression to match date in the format YYYY/MM/DD
    date_pattern = r'\b\d{4}/\d{2}/\d{2}\b'

    # Search for date pattern in the URL string
    match = re.search(date_pattern, url)

    if match:
        return match.group()
    else:
        return None

total_pk = []
total_windiff = []

for file in prog:
    # try:
    pred = torch.load(file)
    filename = str(file.stem).split("_")[0]
    
    # labs = json.load(open(os.path.join(labels_root, f"{filename}.json")))
    # if isinstance(emb, dict):
    #     emb = emb['embeddings']
    # filename = file.stem
    labels_path = os.path.join(labels_root, f"{filename}.json")
    # breakpoint()
    if not os.path.exists(labels_path):
        continue
    label_file = json.load(open(labels_path))
    emb_shape = pred['labels'].shape[0]
    labels = torch.zeros(emb_shape)
    labels[-1]=1
    chapters = [math.ceil(i['start_time'] / 10) for i in label_file['chapters'] if i['start_time']!=0]
    if chapters[-1]==emb_shape:
        chapters = chapters[:-1]
    labels[chapters]=1
    predictions = pred['labels']

    prediction_boundary = get_seg_boundaries(predictions)
    label_boundary = get_seg_boundaries(labels)
    train_pk, _ = pk(prediction_boundary, label_boundary)
    train_windiff, _ = win_diff(prediction_boundary, label_boundary)

    # breakpoint()

    output = {
        "labels": labels,
        "predictions": predictions,
        "pk_score": train_pk,
        "windiff_score": train_windiff
    }

    total_pk.append(train_pk)
    total_windiff.append(train_windiff)

    torch.save(output, f"{output_root}{filename}.pt")
    # except Exception as e:
    #     print(file, e)
        # breakpoint()

print(f"Train pk: {sum(total_pk) / len(total_pk)}")
print(f"Train windiff: {sum(total_windiff) / len(total_windiff)}")