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
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            # nn.BatchNorm1d(lin1_size),
            nn.Dropout(dropout),
            nn.GELU(),

            nn.Linear(lin1_size, lin2_size),
            # nn.BatchNorm1d(lin2_size),
            nn.Dropout(dropout),
            nn.GELU(),
            
            nn.Linear(lin2_size, lin3_size),
            # nn.BatchNorm1d(lin3_size),
            nn.GELU(),
            nn.Linear(lin3_size, num_classes),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        s, b, e = batch.shape
        # x = batch.view(b, s, e)
        out = self.model(batch)
        return out


class MyLSTM1(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int=2, dropout: float = 0.2, num_layers: int = 4) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(512, num_classes)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor):
#         print(f"{x.shape=}")
        l1, h1 = self.lstm1(x)
#         print(f"{l1.shape=}, {h1[0].shape}")
        z0 = self.act1(l1)
#         print(f"{z0.shape=}")
        out = self.linear(z0)
#         print(f"{out.shape=}")
#         out = self.model(x)
#         bs, x, classes = out.shape
        return out


class MyLSTM2(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes:int=2, dropout: float = 0.2, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512, dropout=dropout, num_layers=num_layers)
        self.lstm2 = nn.LSTM(512, 128, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(128, num_classes)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # s, b, e = x.shape
        # x = x.view(b, s, e)
        l1, h1 = self.lstm1(x)
        z0 = self.act1(l1)
        l2, h2 = self.lstm2(z0)
        z0 = self.act1(l2)
        out = self.linear(z0)
        return out #.transpose(1,0)


class MyLSTMBidir(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes:int=2, dropout: float = 0.2, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 512, bidirectional=True, dropout=dropout, num_layers=num_layers)
        # self.lstm2 = nn.LSTM(512*2, 128, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(512*2, num_classes)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # s, b, e = x.shape
        # x = x.view(b, s, e)
        l1, h1 = self.lstm1(x)
        z0 = self.act1(l1)
        # l2, h2 = self.lstm2(z0)
        # z0 = self.act1(l2)
        out = self.linear(z0)
        return out #.transpose(1, 0)


class MySimpleAttn(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes:int=2, dropout: float = 0.2, num_layers: int = 2) -> None:
        super().__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_layers, dropout=dropout, bias=False)
        self.act = nn.GELU()
        self.classification = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        s, b, e = x.shape
        # x = x.view(b, s, e)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        z0, _ = self.attention(q, k, v)
        z0 = self.act(z0)
        out = self.classification(z0)
        return out