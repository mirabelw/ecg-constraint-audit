import torch
from torch import nn

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__();layers=[];previous=12
        for width in [24,32,48,64]:
            layers.extend([nn.Conv1d(previous,width,9,stride=2,padding=4,bias=False),nn.BatchNorm1d(width),nn.ReLU()]);previous=width
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,2))
    def forward(self,x):
        h=self.features(x);return self.head(torch.cat([h.mean(-1),h.amax(-1)],dim=1))

class ResidualBlock(nn.Module):
    def __init__(self, previous, width):
        super().__init__()
        self.branch=nn.Sequential(nn.Conv1d(previous,width,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(width),nn.ReLU(),nn.Conv1d(width,width,7,padding=3,bias=False),nn.BatchNorm1d(width))
        self.skip=nn.Sequential(nn.Conv1d(previous,width,1,stride=2,bias=False),nn.BatchNorm1d(width))
    def forward(self,x):return torch.relu(self.branch(x)+self.skip(x))

class ResidualECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(12,32,11,stride=2,padding=5,bias=False),nn.BatchNorm1d(32),nn.ReLU(),ResidualBlock(32,48),ResidualBlock(48,64))
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,x):
        h=self.features(x);return self.head(torch.cat([h.mean(-1),h.amax(-1)],dim=1))
