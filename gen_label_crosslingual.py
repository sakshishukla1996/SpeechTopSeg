import os 
import torch
from pathlib import Path
from tqdm import tqdm

mono_lingual_path = Path("/data/euronews_dataset/processed_mono_embeddings/")
cross_lingual_path =Path("/data/euronews_dataset/processed_mono_eng_embeddings/")

cross_lingual_files = list(Path(cross_lingual_path).glob(f"**/*.pt"))

for file in tqdm(cross_lingual_files):
    label_path = mono_lingual_path / file.parent.name / file.name
    if not label_path.exists():
        file.unlink()
        continue
    label_file = torch.load(label_path)
    target_file = torch.load(file)
    if type(target_file) is dict:
        continue
    output = {"embeddings": target_file, "labels": label_file["labels"]}
    torch.save(output,file)
