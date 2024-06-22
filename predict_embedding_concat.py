"""
Prediction script for individual embeddings
"""

from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path
import torch
from tqdm import tqdm


from lightning import Callback, LightningDataModule, LightningModule, Trainer


import hydra
from omegaconf import DictConfig

checkpoint = "/data/euronews_dataset/weights/audio/cross_lingual_en/en_en_pt/en_en_pt.ckpt"  # Best checkpoint

# Chnage audio_dataset with transcript embedding dataset for evaluating on transcript.
input_root = "/data/euronews_dataset/audio_dataset/test_data/pt/"

output_root = Path("/data/euronews_dataset/weights/audio/cross_lingual_en/en_en_pt/predictions/")
if not output_root.exists():
    output_root.mkdir(parents=True, exist_ok=True)

language = "pt"

device = torch.device("cuda:0")

# languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
# sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

# s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_deu", device=device)

def main() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global input_root, output_root

    files = list(Path(input_root).glob("*.pt"))

    model_ckpt = torch.load(checkpoint)
    net = model_ckpt['hyper_parameters']["net"]

    net.load_state_dict(model_ckpt['state_dict'], strict=False)
    state_dict = model_ckpt['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    prog = tqdm(files)
    for file in prog:
        data = torch.load(file)
        # If this crashes, change embedding  to embeddings.
        x = data['embeddings']
        
        if isinstance(x, list):
            x = torch.concat(x).to(device)
        else:
            x = x.to(device)
        # breakpoint()
        prepad = torch.cat([torch.zeros(2, x.shape[-1], device=x.device), x])
        currpad = torch.cat([torch.zeros(1, x.shape[-1], device=x.device), x, torch.zeros(1, x.shape[-1], device=x.device)])
        nextpad = torch.cat([x, torch.zeros(2, x.shape[-1], device=x.device)])

        inp = torch.cat([prepad, currpad, nextpad], dim=-1)[:-2]  # 100

        # y = data['labels'].to(device)

        with torch.no_grad():
            logits = net(inp.unsqueeze(0))
        preds = logits.detach().cpu().sigmoid().argmax(-1).squeeze(0)
        preds[-1]=1
        pred_index = (preds==1).nonzero().view(-1)
        # timestamps = pred_index*10
        # timestamps_mins = [[(i//60).item(), (i%60).item()] for i in timestamps]
        output = {
            "filename": file.stem,
            "logits": logits,
            "labels": preds,
            # "timestamps": timestamps,
            # "timestamps_mins": timestamps_mins,
        }
        
        torch.save(output, output_root / f"{file.stem}_predictions.pt")
    print("Saved!")

if __name__=="__main__":
    main()