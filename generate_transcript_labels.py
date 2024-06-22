"""
This script runs the label generation process post transcribing.
"""

from pathlib import Path
import json
import os
import torch
from tqdm import tqdm

transcript_root = Path("/data/euronews_dataset/transcripts/")
embedding_root = "/data/euronews_dataset/processed_mono_embeddings/"
transcripts_output = Path("/data/euronews_dataset/transcripts_labelled/")

def clean_adjacent_duplicates(input_list):
    cleaned_list = []
    previous_number = None
    
    for number in input_list:
        if number != previous_number:
            cleaned_list.append(number)
            previous_number = number
    
    return cleaned_list

prog = tqdm(list(transcript_root.glob("**/*.json")))

for file in prog:
    try:
        data = json.load(open(file, "r"))
    except:
        print(f"Issue in file: {file}")
        continue
    embedding = torch.load(os.path.join(embedding_root, file.parent.name, f"{file.stem}.pt"))
    if isinstance(embedding, torch.Tensor):
        print(f'Error in file: {file.parent.name} / {file.name}')
        continue
    chapters = embedding['chapters']
    # print(chapters)
    speakers = data['speakers']
    data['chapters'] = chapters
    txt=""
    speaker=[]
    speaker_sent = []
    data['sentence']=[]
    data['labels']=[]
    sentence = ""
    # breakpoint()
    for chunk in speakers:
        # try:
        if chunk['timestamp'][0]!=None and chunk['timestamp'][1]!=None:
            chunk_start, chunk_end = chunk['timestamp']
        if chapters!=[]:
            chapter_start, chapter_end = chapters[0]['start_time'], chapters[0]['end_time']
        if (chapter_end > chunk_start and chapter_end <= chunk_end) or chapter_end < chunk_start:
            # there's a split. 
            txt+= chunk['text']
            speaker.append(chunk['speaker'])
            sentence+=chunk['text']
            speaker_sent.append(
                    {
                        "speaker": chunk['speaker'],
                        "sentence": sentence
                    }
                )
            data['sentence'].append(speaker_sent)
            data['chapter_text'] = txt
            labels = [0]*len(speaker_sent)
            labels[-1]=1
            data['labels'].extend(labels)
            # print('Chapter ends here')
            # del chapter[0]
            del chapters[0]

            # clean speaker list
            speaker = []
            speaker_sent=[]
            # clean the appending text
            txt=""
            sentence=""
        else:
            # append the sent in the overall text.
            txt+= chunk['text']
            speaker.append(chunk['speaker'])
            sentence += chunk['text']
            if sentence[-1] in [".", "?", "!"]:
                speaker_sent.append(
                    {
                        "speaker": chunk['speaker'],
                        "sentence": sentence
                    }
                )
                sentence=""
    save_path = ""
    transcript_output_folder = (transcripts_output / file.parent.name)
    transcript_output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saved: {os.path.join(transcript_output_folder, file.name)}")
    with open(os.path.join(transcript_output_folder, file.name), "w") as f:
        json.dump(data, f)