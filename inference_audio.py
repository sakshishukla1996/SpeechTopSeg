from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path
import torch
from torch import nn
import torchaudio
from pytube import YouTube

import pandas as pd

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

from nltk import sent_tokenize
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

import hydra
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datamodule import CustomDataModule
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    get_seg_boundaries
)

log = RankedLogger(__name__, rank_zero_only=True)

url = "https://www.youtube.com/watch?v=dHwNTCLz5ik"  # Make none for local files

input_root = Path("/disk1/projects/sonar_multilingual_segmentation/data")
# url = "https://www.youtube.com/watch?v=X3Po8GPHqDM"

# checkpoint = "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_audio_linear_random-v5.ckpt"
checkpoint = "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_audio_linear_random-v2.ckpt"  # Best checkpoint
output_root = Path("/disk1/projects/sonar_multilingual_segmentation/predictions")
language = "de"
device = torch.device("cuda")

languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_deu", device=device)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global input_root, output_root
    if url is not None:
        yt = YouTube(url)
        input_root /= "temp"
        # breakpoint()
        input_root.mkdir(parents=True, exist_ok=True)
        stream = yt.streams[0]
        download_path = stream.download(output_path =input_root)
        os.system(f"ffmpeg -i \"{download_path}\" -vn {input_root / 'audio.wav'}")
        

    files = list(Path(input_root).glob("*.wav"))
    
    model_ckpt = torch.load(checkpoint)
    net = model_ckpt['hyper_parameters']["net"]
    module = hydra.utils.instantiate(cfg.model, net=net)
    module.load_state_dict(model_ckpt['state_dict'], strict=False)
    module.to(device)
    module.eval()
    
    for file in files:
        audio, sr = torchaudio.load(file)
        audio = audio[:1, :]  # Rechannel
        # Resample to 16k
        audio_low = torchaudio.functional.resample(audio[:1,:], orig_freq=sr, new_freq=16000)
        # Chunk into 10sec
        audio_splits = list(torch.split(audio, split_size_or_sections=(sr * 10), dim=-1))
        audio_10_splits = list(torch.split(audio_low, split_size_or_sections=(16_000 * 10), dim=-1))

        with torch.no_grad():
            embeddings = s2vec_model.predict(audio_10_splits, batch_size=16)
            output_10 = torch.nan_to_num(embeddings, nan=0.0)
            logits = module(output_10.unsqueeze(0))
        preds = logits.softmax(-1).argmax(-1).squeeze(0)

        out = []
        for spl, pred in zip(audio_splits, preds):
            if pred==1:
                out.append(torch.ones((1, 24_000)))
                out.append(spl)
            else:
                out.append(spl)
        print(preds)
        torchaudio.save(output_root / file.name, torch.cat(out, dim=1), sample_rate=sr, bits_per_sample=16)
        print("Saved!")
    
if __name__=="__main__":
    main()