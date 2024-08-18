from pathlib import Path
import os
from tqdm import tqdm
import json

train_files = Path("/data/euronews_dataset/transcript_dataset/training_data/en/")
validation_files = Path("/data/euronews_dataset/transcript_dataset/val_data/en/")

wiki_train_path = Path("/data/euronews_dataset/wiki_format_for_koomri/training_data/en/")
wiki_val_path = Path("/data/euronews_dataset/wiki_format_for_koomri/val_data/en/")


# Train files
prog = tqdm(list(train_files.glob("*.json")))
for file in prog:
    transcript = json.load(open(file, "r"))
    segments = [[j['sentence'] for j in i] for i in transcript['sentence']]
    output_sent = []
    for segment in segments:
        output_sent.append("========,1,chapter.")
        for sent in segment:
            output_sent.append(sent)
    output_text = "\n".join(output_sent)
    if not wiki_train_path.exists():
        wiki_train_path.mkdir(parents=True, exist_ok=True)
    open(os.path.join(wiki_train_path, file.stem), "w").write(output_text)

# Train files
prog = tqdm(list(validation_files.glob("*.json")))
for file in prog:
    transcript = json.load(open(file, "r"))
    segments = [[j['sentence'] for j in i] for i in transcript['sentence']]
    output_sent = []
    for segment in segments:
        output_sent.append("========,1,chapter.")
        for sent in segment:
            output_sent.append(sent)
    output_text = "\n".join(output_sent)
    if not wiki_val_path.exists():
        wiki_val_path.mkdir(parents=True, exist_ok=True)
    open(os.path.join(wiki_val_path, file.stem), "w").write(output_text)