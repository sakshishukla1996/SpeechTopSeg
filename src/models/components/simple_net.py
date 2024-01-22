import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        lin1_size: int = 512,
        lin2_size: int = 256,
        lin3_size: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            # nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            # nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            # nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, num_classes),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        s, b, e = batch.shape
        x = batch.view(b, s, e)
        out = self.model(x)
        return out

class MyLSTM1(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int=2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512)
        self.linear = nn.Linear(512, num_classes)
        self.act1 = nn.ReLU()

        # self.model = nn.Sequential(
        #     nn.LSTM(input_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x: torch.Tensor):
        # breakpoint()
        s, b, e = x.shape
        x = x.view(b, s, e)
        l1, h1 = self.lstm1(x)
        z0 = self.act1(l1)
        out = self.linear(z0)
        # out = self.model(x)
        return out


class MyLSTM2(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes:int=2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512)
        self.lstm2 = nn.LSTM(512, 128)
        self.linear = nn.Linear(128, num_classes)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        s, b, e = x.shape
        x = x.view(b, s, e)
        l1, h1 = self.lstm1(x)
        z0 = self.act1(l1)
        l2, h2 = self.lstm2(z0)
        z0 = self.act1(l2)
        out = self.linear(z0)
        return out
    
class MyLSTMBidir(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes:int=2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512, bidirectional=True)
        self.lstm2 = nn.LSTM(512*2, 128)
        self.linear = nn.Linear(128, num_classes)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        s, b, e = x.shape
        x = x.view(b, s, e)
        l1, h1 = self.lstm1(x)
        z0 = self.act1(l1)
        l2, h2 = self.lstm2(z0)
        z0 = self.act1(l2)
        out = self.linear(z0)
        return out
    