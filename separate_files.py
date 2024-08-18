from pathlib import Path
import os
from tqdm import tqdm


file_root = Path("/data/euronews_dataset/audio_dataset/")
training_files = [Path(f) for f in open(file_root/'training_data.txt', "r").readlines()]
validation_files = [Path(f) for f in open(file_root/'validation_data.txt', "r").readlines()]

transcript_root = "/data/euronews_dataset/transcripts_labelled/"
transcript_embedding_root = "/data/euronews_dataset/transcripts_labelled_embedding/"

prog = tqdm(training_files)
for file in prog:
    prog.set_description(f"Train: {file.parent.name}")
    
    transcript_out = Path("/data/euronews_dataset/transcript_dataset")
    (transcript_out / 'training_data' / file.parent.name).mkdir(parents=True, exist_ok=True)
    embedding_out = Path("/data/euronews_dataset/transcript_embedding_dataset")
    (embedding_out / 'training_data' / file.parent.stem).mkdir(parents=True, exist_ok=True)
    os.system(f"cp {os.path.join(transcript_root, file.parent.name, f'{file.stem}.json')} {transcript_out / 'training_data' / file.parent.name / f'{file.stem}.json'}") # copy transcript files
    os.system(f"cp {os.path.join(transcript_embedding_root, file.parent.name, f'{file.stem}.pt')} {embedding_out / 'training_data' / file.parent.name / f'{file.stem}.pt'}") # copy transcript embeddings

prog = tqdm(validation_files)
for file in prog:
    prog.set_description(f"Val: {file.parent.name}")
    (transcript_out / 'val_data' / file.parent.name).mkdir(parents=True, exist_ok=True)
    (embedding_out / 'val_data' / file.parent.name).mkdir(parents=True, exist_ok=True)
    os.system(f"cp {os.path.join(transcript_root, file.parent.name, f'{file.stem}.json')} {transcript_out / 'val_data' / file.parent.name / f'{file.stem}.json'}") # copy transcript files
    os.system(f"cp {os.path.join(transcript_embedding_root, file.parent.name, f'{file.stem}.pt')} {embedding_out / 'val_data' / file.parent.name / f'{file.stem}.pt'}") # copy transcript embeddings
