import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet

class WDCNN(nn.Module):
    def __init__(self,d_rate):
        super(WDCNN, self).__init__()
        self.first_cnn = nn.Conv1d(1,16,64)
        self.cnn_net = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.FC_net = nn.Sequential(
            nn.Conv1d(64*3,100,1,bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(d_rate),
            nn.Conv1d(100,10,1)
        )
    
    def forward(self, x):
        