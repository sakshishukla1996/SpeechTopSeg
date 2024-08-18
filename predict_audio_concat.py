"""
Prediction script for individual embeddings
"""

from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm


# from lightning import Callback, LightningDataModule, LightningModule, Trainer

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

# import hydra
# from omegaconf import DictConfig

# checkpoint = "/data/euronews_dataset/weights/audio/multilingual_audio/all_all_all/all_all_all.ckpt"  # Best checkpoint
checkpoint = "/data/euronews_dataset/weights/audio/monolingual_audio/en_en_en/en_en_en.ckpt"  # Best checkpoint

# Chnage audio_dataset with transcript embedding dataset for evaluating on transcript.
input_root = "/data/akashvani/orig_videos/"

output_root = Path("/data/akashvani/en_en_en_predictions")
if not output_root.exists():
    output_root.mkdir(parents=True, exist_ok=True)

language = "hi"

device = torch.device("cuda:0")

languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese", "hi": "hindi"}
sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn", "hi": "hin_Deva"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_hin", device=device)

def main() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global input_root, output_root, s2vec_model

    files = list(Path(input_root).glob("**/*.wav"))

    model_ckpt = torch.load(checkpoint)
    net = model_ckpt['hyper_parameters']["net"]

    # net.load_state_dict(model_ckpt['state_dict'], strict=False)
    state_dict = model_ckpt['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()

    prog = tqdm(files)
    for file in prog:
        audio, sr = torchaudio.load(file)
        if sr!=16000:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
            sr = 16000
        audio_split = list(audio.split(sr*10, dim=-1))
        audio_split = [torch.nn.functional.pad(i, (0, sr*10-i.shape[-1])) for i in audio_split]
        # breakpoint()
        if isinstance(audio_split, list):
            x = torch.concat(audio_split).to(device)
        else:
            x = x.to(device)
        with torch.inference_mode():
            audio_embeddings = s2vec_model.predict(audio_split, batch_size=32, n_parallel=2, progress_bar=True)
        x = audio_embeddings
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
        timestamps = pred_index*10
        timestamps_mins = [[(i//60).item(), (i%60).item()] for i in timestamps]
        output = {
            "filename": file.stem,
            "logits": logits,
            "labels": preds,
            "timestamps": timestamps,
            "timestamps_mins": timestamps_mins,
        }
        
        torch.save(output, output_root / f"{file.stem}_predictions.pt")
    print("Saved!")

if __name__=="__main__":
    main()