import os
from pathlib import Path
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

embedding_root = files = Path("/data/euronews_dataset/processed_mono_embeddings/")
training_root = "/data/euronews_dataset/audio_dataset/training_data"
validation_root = "/data/euronews_dataset/audio_dataset/val_data/"

for lang in ["it", "de", "en", "es", "fr", "pt", "ru"]:
    files = (embedding_root / lang).glob("*.pt")
    date_map = []
    for file in files:
        date_map.append((file.stem, torch.load(file)['date']))
    years = [i[1].split('/')[0] for i in date_map]
    print(f"unique years in {lang=} are {Counter(years)}")
    files = sorted(date_map, key=lambda x: x[1])
    train_files, val_files = train_test_split(files, train_size=0.9, shuffle=False)

    training_path = os.path.join(training_root, lang)
    validation_path = os.path.join(validation_root, lang)
    
    if not os.path.exists(training_path):
        Path(training_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(validation_path):
        Path(validation_path).mkdir(parents=True, exist_ok=True)

    train_prog = tqdm(train_files)
    train_prog.set_description(lang)
    for file in train_prog:
        os.system(f"cp {os.path.join(embedding_root, lang, f'{file[0]}.pt')} {os.path.join(training_path, f'{file[0]}.pt')}")

    val_prog = tqdm(val_files)
    val_prog.set_description(lang)
    for file in val_prog:
        os.system(f"cp {os.path.join(embedding_root, lang, f'{file[0]}.pt')} {os.path.join(validation_path, f'{file[0]}.pt')}")

# unique years in lang='it' are Counter({'2023': 563, '2022': 380, '2024': 14})
# it: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 861/861 [00:03<00:00, 265.11it/s]
# it: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 224.68it/s]
# unique years in lang='de' are Counter({'2023': 726, '2022': 347, '2024': 16})
# de: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 980/980 [00:03<00:00, 308.08it/s]
# de: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 109/109 [00:00<00:00, 379.49it/s]
# unique years in lang='en' are Counter({'2022': 596, '2023': 532, '2021': 95, '2024': 75})
# en: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1168/1168 [00:04<00:00, 281.74it/s]
# en: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 130/130 [00:00<00:00, 266.90it/s]
# unique years in lang='es' are Counter({'2019': 144, '2021': 130, '2020': 115, '2022': 102, '2023': 22, '2024': 3, '2018': 1})
# es: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 465/465 [00:01<00:00, 337.38it/s]
# es: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:00<00:00, 291.52it/s]
# unique years in lang='fr' are Counter({'2023': 631, '2022': 614, '2021': 84, '2024': 9})
# fr: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1204/1204 [00:03<00:00, 349.06it/s]
# fr: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 274.83it/s]
# unique years in lang='pt' are Counter({'2022': 300, '2023': 18})
# pt: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 286/286 [00:00<00:00, 304.77it/s]
# pt: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 334.92it/s]
# unique years in lang='ru' are Counter({'2023': 3})
# ru: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 357.75it/s]
# ru: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 227.95it/s]