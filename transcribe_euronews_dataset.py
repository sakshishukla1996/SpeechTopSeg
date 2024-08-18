"""
This script diarizes and transcribes the entire dataset. 
Just insert paths containing all the wav files.  
"""

import os
from pathlib import Path
import json
import torch
from tqdm import tqdm
import math

from transformers import pipeline

# from insanely_fast_whisper.utils.diarize import post_process_segments_and_transcripts, diarize_audio, \
#     preprocess_inputs


# Args #
class Args:
    model_name = "openai/whisper-large-v3"
    device_id ="0"
    flash_attn = False
    hf_token = "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"
    batch_size = 32
    language = None
    task = "transcribe"
    timestamp = "chunk"
    ts = "word" if timestamp == "word" else "chunk"
    generate_kwargs = {"task": task, "language": language}
    diarization_model="pyannote/speaker-diarization-3.1"
    num_speakers=None
    min_speakers=1
    max_speakers=100

    file_name = None

args = Args()
########

wav_root = "/data/tagesschau/more_data/"
labels_output_root = "/data/tagesschau/more_data/"

embedding_files = list(Path(wav_root).glob("**/audio.wav"))
prog = tqdm(embedding_files)

pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"} if args.flash_attn else {"attn_implementation": "sdpa"},
    )

# breakpoint()

for file in prog:
    args.file_name = str(file)

    filename = file.stem
    labels_out = os.path.join(labels_output_root, file.parent.stem)
    Path(labels_out).mkdir(parents=True, exist_ok=True)
    labels_save_path = os.path.join(labels_out, f"transcripts_ifw.json")
    if os.path.exists(labels_save_path):
        continue
    os.system(f"insanely-fast-whisper --file-name {str(file)} --transcript-path {labels_save_path} --flash False --hf_token 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 --diarization_model pyannote/speaker-diarization-3.1 2>/tmp/null")
    prog.set_postfix({"file": file})