import os
from pathlib import Path
from tqdm import tqdm

audio_dataset = Path("/data/euronews_dataset/audio_dataset/training_data")
# audio_dataset = Path("/data/euronews_dataset/audio_dataset/val_data")
# audio_dataset = Path("/data/euronews_dataset/audio_dataset/test_data")

inp_path = audio_dataset

cl_root = Path(f"/data/euronews_dataset/audio_dataset_cl/{inp_path.name}")

files = list(Path(inp_path).glob(f"**/*.pt"))

# val_files = list(Path(audio_dataset_val).glob(f"**/*.pt"))
# test_files = list(Path(audio_dataset_test).glob(f"**/*.pt"))

for file in tqdm(files):
    from_path = Path("/data/euronews_dataset/processed_mono_eng_embeddings") / file.parent.name / file.name
    out_path = cl_root / file.parent.name / file.name
    if not from_path.exists():
        continue
    os.system(f"cp {from_path} {out_path}")