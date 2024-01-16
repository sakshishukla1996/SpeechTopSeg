from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random
import os

# from sonar.models.sonar_text import (
#     load_sonar_text_encoder_model,
#     load_sonar_tokenizer,
# )
# from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

class CustomDataset(Dataset):
    def __init__(self, filelist:list[Path], mode:str="seq", cuda:bool=False) -> None:
        super().__init__()
        self.filelist = filelist
        self.stage=mode
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.stage=="random":
            emb, labs = self._get_random_data()
        elif self.stage=="seq":
            file = self.filelist[index]
            emb, labs = self._get_seq_data(file)
        return emb, labs
    
    def _get_seq_data(self, filepath):
        file = filepath
        # print("================================================= filepath ================================================")
        # embs=[]
        # labs=[]
        d=torch.load(file)
        # print(d['embeddings'].shape, d['labels'])
        emb = d['embeddings'].cpu()
        lab = d['labels'].cpu()
        labs = torch.tensor(lab)
        return emb, labs
    
    def _get_random_data(self):
        nfiles=torch.randint(10, 30, (1,)).item()
        embs=[]
        labs=[]
        for file in range(nfiles):
            randid = torch.randint(0, len(self.filelist), (1,)).item()
            d=torch.load(self.filelist[randid])
            embs.append(d['embeddings'].cpu())
            lab = [0]*d['embeddings'].shape[0]
            lab[-1]=1
            labs.extend(lab)
        embs = torch.cat(embs)
        labs = torch.tensor(labs)
        return embs, labs

class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        train_filelist: list,
        val_filelist: list,
        # test_filelist: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.data_train: Optional[Dataset] = CustomDataset(train_filelist, mode="random")
        self.data_val: Optional[Dataset] = CustomDataset(val_filelist, mode="seq")
        # self.data_test: Optional[Dataset] = CustomDataset(val_filelist, stage="test")

        self.batch_size_per_device = batch_size


    # def prepare_data(self) -> None:
    #     pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=0, #self.hparams.num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=0, #self.hparams.num_workers,
            pin_memory=False,
            shuffle=False,
        )

    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the test dataloader.

    #     :return: The test dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size_per_device,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #     )
#===========================================================================================
    
class CustomDataset2(Dataset):
    def __init__(self, filelist:list[Path], mode:str="seq", nfiles:int=1, cuda:bool=False, *args, **kwargs) -> None:
        super().__init__()

        self.english = list(Path("/data/preproc_sonar_single/en/").glob("*.pt"))
        print([i for i in self.english if not os.path.exists(i)])
        print(f"Number of files are {len(self.english)=}")
        self.german = list(Path("/data/preproc_sonar_single/de").glob("*.pt"))
        print([i for i in self.german if not os.path.exists(i)])
        print(f"Number of files are {len(self.german)=}")
        self.spanish = list(Path("/data/preproc_sonar_single/es").glob("*.pt"))
        print([i for i in self.spanish if not os.path.exists(i)])
        print(f"Number of files are {len(self.spanish)=}")
        self.french = list(Path("/data/preproc_sonar_single/fr").glob("*.pt"))
        print([i for i in self.french if not os.path.exists(i)])
        print(f"Number of files are {len(self.french)=}")
        self.portu = list(Path("/data/preproc_sonar_single/pt").glob("*.pt"))
        print([i for i in self.portu if not os.path.exists(i)])
        print(f"Number of files are {len(self.portu)=}")
        
        self.stage=mode
        self.nfiles=nfiles
        self.counts = 0
    
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        if self.stage=="random":
            emb, labs = self._get_random_data()
        elif self.stage=="seq":
            file = self.filelist[index]
            emb, labs = self._get_seq_data(file)
        # else:
        #     file = self.filelist[index]
        #     emb, labs = self._get_embeddings(file)
        return emb, labs
    
    def _get_seq_data(self, filepath):
        file = filepath
        # print("================================================= filepath ================================================")
        # embs=[]
        # labs=[]
        d=torch.load(file)
        # print(d['embeddings'].shape, d['labels'])
        emb = d['embeddings'].cpu()
        lab = d['labels'].cpu()
        labs = torch.tensor(lab)
        return emb, labs
    
    def _get_random_data(self):
        sel_lang = random.choice([self.english, self.german, self.spanish, self.french, self.portu])
        # files = len(sel_lang)
        # idxs = torch.randint(0, len(sel_lang), (self.nfiles,)).tolist()
        # selected = files[idxs]
        embs=[]
        labs=[]
        for file in range((self.nfiles%1500)+1):
            randid = torch.randint(0, len(sel_lang), (1,)).item()
            d=torch.load(sel_lang[randid])
            embs.append(d['embeddings'].cpu())
            lab = [0]*d['embeddings'].shape[0]
            lab[-1]=1
            labs.extend(lab)
        embs = torch.cat(embs)
        labs = torch.tensor(labs)
        if self.counts==1000:
            self.counts=0
            self.nfiles+=1
            print(f"Increased the number if files to {self.nfiles}")
        else:
            self.counts+=1
        return embs, labs

class CustomDataModule2(LightningDataModule):
    def __init__(
        self,
        train_filelist: list,
        val_filelist: list,
        # test_filelist: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.data_train: Optional[Dataset] = CustomDataset2(train_filelist, mode="random", nfiles=2)
        self.data_val: Optional[Dataset] = CustomDataset(val_filelist, mode="seq")
        # print(f"Current number of files being processed: {2**self.current_epoch}")
        # self.data_test: Optional[Dataset] = CustomDataset(val_filelist, stage="test")

        self.batch_size_per_device = batch_size


    # def prepare_data(self) -> None:
    #     pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=0, #self.hparams.num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=0, #self.hparams.num_workers,
            pin_memory=False,
            shuffle=False,
        )
