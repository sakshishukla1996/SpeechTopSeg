from pathlib import Path
import torch
import hydra
from tqdm import tqdm
from src.utils import get_seg_boundaries, pk, win_diff


wiki_root = f"/data/euronews_dataset/wiki_format_for_koomri/test_data/"
files = list(Path(wiki_root).glob("*.pt"))

prog = tqdm(files)
checkpoint = "/data/euronews_dataset/weights/transcript_linear_4layer.ckpt"
device = torch.device("cuda:0")

model_ckpt = torch.load(checkpoint)
net = model_ckpt['hyper_parameters']["net"]
state_dict = model_ckpt['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('net.', '')] = state_dict.pop(key)

net.load_state_dict(state_dict, strict=False)
net.to(device)
net.eval()

pks = []
windiffs= []

for file in prog:
    d = torch.load(file)
    emb = d['embeddings'].unsqueeze(0).to(device)
    labels = d['labels']
    with torch.inference_mode():
        output = net(emb)[0].detach().cpu().sigmoid().argmax(-1).tolist()
        output[-1]=1
    prediction_boundary = get_seg_boundaries(output)
    label_boundary = get_seg_boundaries(labels)
    train_pk, _ = pk(prediction_boundary, label_boundary)
    train_windiff, _ = win_diff(prediction_boundary, label_boundary)
    pks.append(train_pk)
    windiffs.append(train_windiff)

pk = torch.tensor(pks).mean()
windiff = torch.tensor(windiffs).mean()
print(f"Pk score is {pk=}")
print(f"Windiff score is {windiff=}")
