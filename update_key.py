import os
import torch
from pathlib import Path
from tqdm import tqdm

# audio_dataset = Path("/data/euronews_dataset/audio_dataset_cl/training_data")
# audio_dataset = Path("/data/euronews_dataset/audio_dataset_cl/val_data")
audio_dataset = Path("/data/euronews_dataset/audio_dataset_cl/test_data")

files = list(Path(audio_dataset).glob(f"**/*.pt"))
for file in tqdm(files):
    data = torch.load(file)
    data['embeddings'] = data.pop('embbeddings')
    torch.save(data,file)