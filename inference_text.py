from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path
import torch
from torch import nn
from pprint import pprint

import pandas as pd

from sonar.models.sonar_text import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
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


input_root = "/disk1/projects/sonar_multilingual_segmentation/data"
# checkpoint = "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_en_bilstm_random.ckpt"
checkpoints = [
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_en_linear_wiki727.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_en_attention_random_dw-v3.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_en_lstmbidir_wiki727.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_en_linear_random-v2.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/sonar_dw_multihead16.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v20.ckpt", # A good checkpoint
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v22.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v24.ckpt",
#     "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v26.ckpt",
    # "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v23.ckpt",
    # "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v22.ckpt",
    # "/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v32.ckpt"
    ]
checkpoints = [f"/disk1/projects/sonar_multilingual_segmentation/checkpoints/sonar/last-v{i}.ckpt" for i in range(10,34)]
output_root = Path("/disk1/projects/sonar_multilingual_segmentation/predictions")
language = "en"
device = "cuda"
thresh = 0.3

languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
device = torch.device("cuda")
t2enc = load_sonar_text_encoder_model("text_sonar_basic_encoder", device=device).eval()
text_tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")

embedder = TextToEmbeddingModelPipeline(t2enc, text_tokenizer, device=device)

def get_random_background():
    color = torch.randint(150, 255, (3,)).tolist()
    hex = "#%02x%02x%02x" % (color[0], color[1], color[2])
    return hex

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    jsonl = False
    files = list(Path(input_root).glob("*.jsonl"))
    if len(files)!=0:
        jsonl=True
    else:
        files = list(Path(input_root).glob("*.txt"))
    
    for file in files:
        if jsonl:
            df = pd.read_json(file, lines=True)
            text = []
            labels = []
            for isx, sent in enumerate(df['sourceItemMainText']):
                if len(sent.strip())==0:
                    continue
                s = sent_tokenize(sent, language=languages[language])
                text+=s
                try:
                    lab = [0]*len(s)
                    lab[-1]=1
                    labels.extend(lab)
                except:
                    breakpoint()
        else:
            readdata = open(file, "r").read()
            text = sent_tokenize(readdata, language=languages[language])
        text = [i[:512] for i in text]
        with torch.no_grad():
            embeddings = embedder.predict(text, source_lang=sonar_lang[language], batch_size=64)
        

        outs = []
        for checkpoint in checkpoints:
            model_ckpt = torch.load(checkpoint)
            net = model_ckpt['hyper_parameters']["net"]
            module = hydra.utils.instantiate(cfg.model, net=net)
            module.load_state_dict(model_ckpt['state_dict'], strict=False)
            module.to(device)
            module.eval()
            with torch.no_grad():
                logits = module(embeddings.unsqueeze(0))
            # out = nn.functional.softmax(logits, -1).squeeze(1).argmax(-1).detach().cpu()
            out = logits.softmax(-1).argmax(-1).squeeze(0).detach().cpu()
            outs.append(out)
        print(file)
        pprint(outs)
        print("-"*80)
        pred = torch.stack(outs).to(torch.float).mean(0)
        pred[pred >= thresh] = 1
        pred[pred < thresh] = 0
        # pred = pred.to("torch.int")
        print(file.stem, out)
        # breakpoint()
        if jsonl:
            result = ["<p>", f'<mark style="background: {get_random_background()}!important">']
            for l, o, se in zip(labels, out, text):
                if l==1:
                    result.append("</p> <p>")
                if o==1:
                    result.append(f'</mark> <mark style="background: {get_random_background()}!important">')
                result.append(se)
            result.append("</mark> </p>")
            with open(output_root / f"{file.stem}.html", "w") as f:
                f.write(" ".join(result))
        else:
            result = ["<p>", f'<mark style="background: {get_random_background()}!important">']
            for o, se in zip(out, text):
                if o==1:
                    result.append("</mark> </p>")
                    result.append("<p>")
                    result.append(f'<mark style="background: {get_random_background()}!important">')
                result.append(se)
            result.append("</p>")
            with open(output_root / f"{file.stem}.md", "w") as f:
                f.write(" ".join(result))


if __name__=="__main__":
    main()