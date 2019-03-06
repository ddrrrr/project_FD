import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
from multiprocessing.dummy import Pool as ThreadPool

class WDCNN_Net(nn.Module):
    def __init__(self,d_rate=0.5):
        super(WDCNN_Net, self).__init__()
        self.first_cnn = nn.Conv1d(1,16,64,8,28)
        self.cnn_net = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,1,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,3,1,1),
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
        x = x.view(x.size(0), -1, 1)
        out = self.FC_net(x)
        out = out.view(out.size(0), -1)
        return out

class WDCNN():
    def __init__(self):
        self.piece_length = 2048
        self.dataset = DataSet.load_dataset(name='cwru_data')
        self.lr = 0.015
        self.epochs = 1000
        self.batches = 100
        self.batch_size = 100

    def train(self):
        train_data,train_label = self._preprocess('3hp')
        test_data,test_label = self._preprocess('2hp')
        wdcnn = WDCNN_Net().cuda()
        wdcnn.apply(self._weights_init)
        optimizer = optim.Adam(wdcnn.parameters(), lr=self.lr)

        for e in range(1, self.epochs+1):
            train_loss, train_acc = self._fit(e, wdcnn, optimizer, [train_data,train_label])
            val_loss, val_acc = self._evaluate(wdcnn, [test_data,test_label])
            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e][train_acc:%.2f%%][val_acc:%.2f%%]"
                % (e, train_loss, val_loss, train_acc*100, val_acc*100))

    def test(self):
        pass

    def _weights_init(self,m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data,mean=0,std=0.1)
            if isinstance(m.bias, torch.nn.parameter.Parameter):
                nn.init.constant_(m.bias.data, val=0.1)

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

        label_asign = {'normal':0,'B007':1,'B014':2,'B021':3,'IR007':4,'IR014':5,'IR021':6,'OR007':7,'OR014':8,'OR021':9}
        for i,x in enumerate(temp_label):
            for k in label_asign.keys():
                if k in x:
                    temp_label[i] = label_asign[k]
                    break
        
        # r_data,r_label = [],[]
        # for i,x in enumerate(temp_data):
        #     for j in range(round((len(x)-self.piece_length)/self.piece_shift)):
        #         r_data.append(x[j*self.piece_shift:j*self.piece_shift+self.piece_length])
        #         r_label.append(temp_label[i])

        return temp_data, temp_label

    def _fit(self, e, model, optimizer, train_iter, drop_rate=[0.1,0.9]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        label_all = [()]*10
        for i,x in enumerate(train_iter[1]):
            label_all[x] += (i,)

        for batch in range(self.batches):
            label = np.random.randint(0,10,size=self.batch_size)
            random_idx = [random.choice(label_all[x]) for x in label]
            label = label.reshape(1,-1)
            
            random_start = [random.randint(0,len(train_iter[0][x])-self.piece_length) for x in random_idx]
            data = [train_iter[0][random_idx[i]][random_start[i]:random_start[i]+self.piece_length].reshape(1,-1) for i in range(self.batch_size)]

            data, label = np.concatenate(data,axis=0), np.concatenate(label,axis=0)
            data = (data - np.repeat(np.min(data,axis=1,keepdims=True),self.piece_length,axis=1)) / \
                    np.repeat(np.max(data,axis=1,keepdims=True)-np.min(data,axis=1,keepdims=True),self.piece_length,axis=1)
            data = data[:,np.newaxis,:]
            label = label.reshape(-1,)
            data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
            data, label = data.type(torch.FloatTensor), label.type(torch.LongTensor)
            data, label = Variable(data).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output = model(data, random.uniform(*drop_rate))
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = F.cross_entropy(output,label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            torch.cuda.empty_cache()        #empty useless variable
        return total_loss / self.batches, correct / total
        
    def _evaluate(self, model, val_iter):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        label_all = [()]*10
        for i,x in enumerate(val_iter[1]):
            label_all[x] += (i,)

        for batch in range(self.batches):
            label = np.random.randint(0,10,size=self.batch_size)
            random_idx = [random.choice(label_all[x]) for x in label]
            label = label.reshape(1,-1)

            random_start = [random.randint(0,len(val_iter[0][x])-self.piece_length) for x in random_idx]
            data = [val_iter[0][random_idx[i]][random_start[i]:random_start[i]+self.piece_length].reshape(1,-1) for i in range(self.batch_size)]

            data, label = np.concatenate(data,axis=0), np.concatenate(label,axis=0)
            data = (data - np.repeat(np.min(data,axis=1,keepdims=True),self.piece_length,axis=1)) / \
                    np.repeat(np.max(data,axis=1,keepdims=True)-np.min(data,axis=1,keepdims=True),self.piece_length,axis=1)
            data = data[:,np.newaxis,:]
            label = label.reshape(-1,)
            data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
            data, label = data.type(torch.FloatTensor), label.type(torch.LongTensor)
            data, label = Variable(data).cuda(), Variable(label).cuda()
            output = model(data, 1)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = F.cross_entropy(output,label)
            total_loss += loss.data
            torch.cuda.empty_cache()        #empty useless variable
        return total_loss / self.batches, correct / total


if __name__ == "__main__":
    process = WDCNN()
    process.train()
    