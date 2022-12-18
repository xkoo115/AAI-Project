import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class EsConvModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EsConvModel, self).__init__()
        self.net = nn.Sequential(
            Block(in_channels, out_channels),
            Block(out_channels, out_channels),
            Block(out_channels, out_channels),
            nn.MaxPool1d(2),
            #nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        return self.net(x)

class Recogniser(nn.Module):
    def __init__(self, in_channels=1, n_feat=16):
        super(Recogniser, self).__init__()
        self.init_net = nn.Conv1d(in_channels, n_feat, 3, 1, 1)
        self.conv = nn.Sequential(
            EsConvModel(n_feat, n_feat*2),
            EsConvModel(n_feat*2, n_feat*2),
            EsConvModel(n_feat*2, n_feat*4),
            EsConvModel(n_feat*4, n_feat*4),
            EsConvModel(n_feat*4, n_feat*8),
            EsConvModel(n_feat*8, n_feat*8),
            EsConvModel(n_feat*8, n_feat*4),
            EsConvModel(n_feat*4, n_feat*4),
            EsConvModel(n_feat*4, n_feat*2),
            nn.Conv1d(n_feat*2, 1, 3, 1, 1)
        )
        self.linear = nn.Sequential(
            nn.Linear(390, 300),
            nn.ReLU(),
            nn.Linear(300, 250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.init_net(x)
        x = self.conv(x)
        return self.linear(x)

