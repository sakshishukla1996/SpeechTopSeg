"""
Preprocess audio dataset specifically designed for euronews corpus. 
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
import math
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

languages = {"es":"spa", "pt":"por", "en":"eng", "ru":"rus", "fr":"fra", "de":"deu", "it":"ita"}
lang = "en"
language = "pt"
input_root = f"/data/euronews_dataset/processed_mono/{language}"
input_ext = "webm"
files = list(Path(input_root).glob(f"**/*.{input_ext}"))
bs=16

extract_wav = False
extract_emb = True
get_stats = False

if extract_wav:

    output_root = Path("/data/euronews_dataset/processed_mono/")
    output_ext = "wav"
    
    prog = tqdm(files)
    for file in prog:
        filename = file.name
        input = file.parent
        output = Path(os.path.join(output_root, str(input).replace(input_root, str(output_root))))
        output.mkdir(parents=True, exist_ok=True)
        # if os.path.exists(f"{output / filename.replace(input_ext, output_ext)}"):
        #     continue
        # Convert webm to wav
        # os.system(f"ffmpeg -y -i {input / filename} -vn {output / filename.replace(input_ext, output_ext)} -loglevel quiet")

        # Convert wav stereo to mono with 16 bit bit depth
        os.system(f"ffmpeg -y -i {input / filename} -vn -acodec pcm_s16le -ac 1 -ar 16000 {os.path.join(output, filename.replace(input_ext, output_ext))} -loglevel quiet")
        prog.set_postfix({"filename": os.path.join(output, filename.replace(input_ext, output_ext))})

if extract_emb:
    output_root = Path("/data/euronews_dataset/processed_mono_eng_embeddings/")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working with the model: sonar_speech_encoder_{languages[language]} for language {language}")
    mapper = SpeechToEmbeddingModelPipeline(encoder=f"sonar_speech_encoder_{languages[lang]}", device=device)
    files = list(Path(input_root).glob("*.wav"))
    assert len(files)!=0, "Please make sure you are using the correct file folder containing wavs."
    print(len(files))
    progress = tqdm(files)
    for file in progress:
        output = Path(os.path.join(output_root, file.parent.name))
        output.mkdir(parents=True, exist_ok=True)
        output_path = Path(os.path.join(output, file.name.replace('wav', 'pt')))        
        audio, sr = torchaudio.load(file)
        assert sr==16000, "Please check your SR value. Should be strictly 16kHz."
        pad = math.ceil(audio.shape[-1] / 160000) * 160000 - audio.shape[-1]
        audio_low = torch.nn.functional.pad(audio, (0, pad), mode='constant')
        audio_10_splits = list(torch.split(audio_low, split_size_or_sections=(16_000 * 10), dim=-1))
        # batch size
        with torch.no_grad():
            output_10 = mapper.predict(audio_10_splits, batch_size=bs).detach().cpu()
        output_10 = torch.nan_to_num(output_10, nan=0.0)
        torch.save(output_10, output_path)

if get_stats:
    # files = list(Path(input_root).glob(f"**/*.{input_ext}"))
    input_root = f"/data/euronews_dataset/processed_mono/{language}"
    input_ext = "wav"
    files = list(Path(input_root).glob(f"**/*.{input_ext}"))
    prog = tqdm(files)
    samples = {}
    for file in prog:
        try:
            meta = torchaudio.info(file)
            duration = meta.num_frames / meta.sample_rate
            language = file.parent.name
            if samples.get(language) == None:
                samples[file.stem] = []
            samples[file.stem].append(duration)
        except:
            print("Filed on file: ", file)

# breakpoint()