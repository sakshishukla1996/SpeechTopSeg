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
    

class AudioDataset2(Dataset):
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
        # Create 3d input
        prepad = torch.cat([torch.zeros(2, file.shape[-1]), file])
        sidepad = torch.cat([torch.zeros(1, file.shape[-1]), file, torch.zeros(1, file.shape[-1])])
        postpad = torch.cat([file, torch.zeros(2, file.shape[-1])])
        dense_inp = torch.cat([prepad, sidepad, postpad], dim=-1)  # Remove last as there's no information
    
        # Create 3d labels
        prepad = torch.cat([torch.zeros(2), label])
        sidepad = torch.cat([torch.zeros(1), label, torch.zeros(1)])
        postpad = torch.cat([label, torch.zeros(2)])
        dense_lab = torch.stack([sidepad, postpad])

        final_lab = torch.logical_or(dense_lab[0],dense_lab[1]).to(int)  # Remove last as there's no information 
        # final_lab = torch.cat([label, torch.tensor([0])])
        return dense_inp[:-2], final_lab[:-2]
        # breakpoint()

        # rand_idx = torch.randint(1, file.shape[0], (1,)).item()
        # x = dense_inp[rand_idx]
        # y = final_lab[rand_idx]
        # return x, y.item()

    def __len__(self):
        return len(self.files)
    
class EuroNewsDataset(Dataset):
    def __init__(self, embeddings_path: list) -> None:
        super().__init__()
        if isinstance(embeddings_path, list):
            embeddings_path = [embeddings_path]
        self.files = []
        for epth in embeddings_path:
            self.files.extend(glob(epth))
        print(f"Found {len(self.files)=}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        e = torch.load(self.files[idx])
        emb = e['embeddings']
        labs = e['labels']
        return emb, labs

    
class EuroNewsDatasetConcat(Dataset):
    def __init__(self, embeddings_path: list) -> None:
        super().__init__()
        # if isinstance(embeddings_path, list):
        #     embeddings_path = [embeddings_path]
        self.files = []
        for epth in embeddings_path:
            self.files.extend(glob(epth))
        print(f"Found {len(self.files)=}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        e = torch.load(self.files[idx])
        emb = e['embeddings']
        labs = e['labels']
        # labels = label = torch.zeros(emb.shape[0]).type(torch.LongTensor)
        # Create 3d input
        # breakpoint()
        prepad = torch.cat([torch.zeros(2, emb.shape[-1], device=emb.device), emb])
        sidepad = torch.cat([torch.zeros(1, emb.shape[-1], device=emb.device), emb, torch.zeros(1, emb.shape[-1], device=emb.device)])
        postpad = torch.cat([emb, torch.zeros(2, emb.shape[-1], device=emb.device)])
        dense_inp = torch.cat([prepad, sidepad, postpad], dim=-1)  # Remove last as there's no information
    
        # Create 3d labels
        prepad = torch.cat([torch.zeros(2), labs])
        sidepad = torch.cat([torch.zeros(1), labs, torch.zeros(1)])
        postpad = torch.cat([labs, torch.zeros(2)])
        dense_lab = torch.stack([sidepad, postpad])

        final_lab = torch.logical_or(dense_lab[0],dense_lab[1]).to(int)  # Remove last as there's no information 
        # final_lab = torch.cat([label, torch.tensor([0])])
        return dense_inp[:-2], final_lab[:-2]

if __name__=="__main__":
    # ads = AudioDataset2(
    #     filepath="/data/tagesschau/more_data_preprocessed/10sec/*.pt", 
    #     jsonpath="/data/tagesschau/more_data/labels.json", 
    #     window_size=10
    # )
    # p, l = ads[0]
    # print(p.shape, l)
    data = EuroNewsDatasetConcat(embeddings_path=["/data/euronews_dataset/training_data/fr/*.pt"])
    emb, lab = data[0]
    print(emb.shape, lab.shape)