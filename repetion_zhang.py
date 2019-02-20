import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet

class WDCNN_Net(nn.Module):
    def __init__(self,d_rate):
        super(WDCNN_Net, self).__init__()
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
    
    def forward(self, x, drop_rate):
        x = self.first_cnn(x)
        x = F.dropout(x,p=drop_rate)
        x = self.cnn_net(x)
        x = x.view(x.size(0), -1)
        out = self.FC_net(x)
        return out

class WDCNN():
    def __init__(self):
        self.piece_length = 2048
        self.piece_shift = 33
        self.dataset = DataSet.load_dataset(name='cwru_data')

    def train(self):
        pass

    def test(self):
        pass

    def _preprocess(self, select):
        if select == '0hp':
            temp_data = self.dataset.get_value('data',condition={'load':'0'})
            temp_label = self.dataset.get_value('name',condition={'load':'0'})
        elif select == '1hp':
            temp_data = self.dataset.get_value('data',condition={'load':'1'})
            temp_label = self.dataset.get_value('name',condition={'load':'1'})
        elif select == '2hp':
            temp_data = self.dataset.get_value('data',condition={'load':'2'})
            temp_label = self.dataset.get_value('name',condition={'load':'2'})
        elif select == '3hp':
            temp_data = self.dataset.get_value('data',condition={'load':'3'})
            temp_label = self.dataset.get_value('name',condition={'load':'3'})
        elif select == 'all':
            temp_data = self.dataset.get_value('data',condition='all')
            temp_label = self.dataset.get_value('name',condition='all')
        else:
            raise ValueError('paramenter select should be xhp or all!')

        r_data = []
        r_label = []
        for i,data in enumerate(temp_data):
            for 


        temp_data, temp_label = self._shuffle(temp_data,temp_label)
        return temp_data, temp_label
        