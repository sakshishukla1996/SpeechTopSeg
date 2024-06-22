from pathlib import Path
import json
import torch
import os
import sys
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from tqdm import tqdm

languages = {"es":"spa_Latn", "pt":"por_Latn", "en":"eng_Latn", "ru":"eng_Latn", "fr":"fra_Latn", "de":"deu_Latn", "it":"ita_Latn"}

language = "de"
input_root = f"/data/tagesschau/transcript_dataset"
output_root = f"/data/tagesschau/transcript_embedding_dataset"
input_ext = "json"
files = list(Path(input_root).glob(f"**/*.{input_ext}"))
device = "cuda:0"
bs=16


mapper = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)
prog = tqdm(files)

for file in prog:
    prog.set_description(file.parent.name)
    # Do something.
    data = json.load(open(file, "r"))
    segments = [[j['sentence'] for j in i] for i in data['sentence']]
    output_segments = []
    for segment in segments:
        with torch.no_grad():
            output = mapper.predict(segment, batch_size=bs, source_lang=languages[language]).detach().cpu()
        output_segments.append(output)
    embedding_file={
        "sentences": segments,
        "embeddings": output_segments,
        "labels": data['labels']
    }
    output_path = os.path.join(output_root, file.parent.name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    torch.save(embedding_file, os.path.join(output_path, f"{file.stem}.pt"))
    prog.set_postfix({"file": file.name})