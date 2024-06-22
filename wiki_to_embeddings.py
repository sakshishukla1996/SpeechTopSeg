from pathlib import Path
import json
import torch
import os
import sys
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from tqdm import tqdm
import re

languages = {"es":"spa_Latn", "pt":"por_Latn", "en":"eng_Latn", "ru":"eng_Latn", "fr":"fra_Latn", "de":"deu_Latn", "it":"ita_Latn"}

language = "en"
wiki_root = Path("/data/projects/projects/text-segmentation/dataset/wiki_727_debug/test/")
output_root = f"/data/euronews_dataset/wiki_format_for_koomri/test_data/"

Path(output_root).mkdir(parents=True, exist_ok=True)

files = [i for i in list(Path(wiki_root).glob(f"**/*")) if not i.is_dir()]
files = [i for i in files if "106667" in str(i)]
print(f'The number of {len(files)=}')
device = "cuda:0"
bs=16

mapper = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)
prog = tqdm(files)
pattern = r"========,.,([\w ]+)."

for file in prog:
    prog.set_description(file.parent.name)
    # Do something.
    with open(file, "r") as f:
        data = f.readlines()
    output_texts = []
    output_segments = []
    
    for idx, line in enumerate(data):
        if re.match(pattern, line):
            output_texts.append(output_segments)
            output_segments = []
        else:
            output_segments.append(line)
    output_texts = [i for i in output_texts if len(i)!=0]
    all_texts = [j for i in output_texts for j in i]

    labels=[]
    for seg in output_texts:
        labels.extend([0]*len(seg))
        labels[-1]=1
    with torch.no_grad():
        embeddings = mapper.predict(all_texts, source_lang=languages[language], batch_size=bs, max_seq_len=512)
    output = {
        "segments": output_texts,
        "text": "".join(all_texts),
        "embeddings": embeddings,
        "labels": labels,
    }
    torch.save(output, os.path.join(output_root, f"{file.stem}.pt"))