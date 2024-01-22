import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from glob import glob

class AudioDataset(Dataset):
    def __init__(self, filepath, jsonpath, window_size=10):
        if isinstance(filepath, str):
            self.files = [Path(i) for i in glob(filepath, recursive=True)]
        else:
            self.files = list(filepath)
        self.labels = json.load(open(jsonpath, "r"))
        self.window_size = window_size

    def __getitem__(self, idx):
        filepath = self.files[idx]
        file = torch.load(filepath).to("cpu")
        duration = [int((m*60 + s) / self.window_size) for m,s in self.labels[str(filepath.stem)]['duration'] if int((m*60 + s) / self.window_size)<100]
        label = torch.zeros(len(file)).type(torch.LongTensor)
        label[duration[1:]]=1
        label[-1]=1
        return file, label

    def __len__(self):
        return len(self.files)