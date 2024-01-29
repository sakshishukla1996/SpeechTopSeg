import random
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from omegaconf import DictConfig
from glob import glob

class TextDataset(Dataset):
    def __init__(self, filepath: dict, mode:str = "seq", nfiles: int=1, max_files: int = None) -> None:
        """
        filepath: dict {"language": [glob_pattern1, glob_pattern2, file1, file2]}
        
        For Seq mode, please preproc the files.
        """
        self.files = {}
        self.nfiles = nfiles
        self.len_files = 0
        self.max_files = max_files
        self.stage = mode
        self.counts=0
        for k, v in filepath.items():
            self.files[k] = []
            for pattern in v:
                if "*" in pattern:
                    ret_files = glob(pattern)
                    self.files[k].extend(ret_files)
                    self.len_files+=len(ret_files)
                elif isinstance(pattern, str) and "*" not in pattern:
                    self.files[k].append(pattern)
                    self.len_files+=1

    def __len__(self):
        return self.len_files if self.max_files is None else self.max_files


    def __getitem__(self, index):

        if self.stage == "random":
            emb, labs = self._get_random_data()
        elif self.stage == "seq":
            # print(f"================ {index} ================")
            for lang in self.files.keys():
                l = len(self.files[lang])
                if l>index:
                    file = self.files[lang][index]
                else:
                    index -= len(self.files[lang])
                    continue
            # print(f"================ {file} ================")
            emb, labs = self._get_seq_data(file)

        return emb, labs 
    
    def _get_seq_data(self, filepath):
        file = filepath
        # print("=============================== filepath ===============================")
        # embs=[]
        # labs=[]
        d=torch.load(file)
        emb = d['embeddings'].cpu()
        lab = d['labels']
        labs = torch.tensor(lab)
        return emb, labs

    def _get_random_data(self):
        sel_lang = random.choice(list(self.files.keys()))
        embs=[]
        labs=[]
        for file in range((self.nfiles%1500)+1):
            randid = torch.randint(0, len(self.files[sel_lang]), (1,)).item()
            d=torch.load(self.files[sel_lang][randid])
            embs.append(d['embeddings'].cpu())
            lab = [0]*d['embeddings'].shape[0]
            lab[-1]=1
            labs.extend(lab)
        embs = torch.cat(embs)
        labs = torch.tensor(labs)
        if self.counts==5000:
            self.counts=0
            self.nfiles+=1
            print(f"Increased the number if files to {self.nfiles}")
        else:
            self.counts+=1
        return embs, labs