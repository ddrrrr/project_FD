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
        self.first_cnn = nn.Conv1d(1,16,64,16,24)
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
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Conv1d(64,64,3,1,1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.MaxPool1d(2),
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
        self.lr = 1e-3
        self.epochs = 100
        self.batches = 100
        self.batch_size = 8

    def train(self):
        train_data,train_label = self._preprocess('0hp')
        test_data,test_label = self._preprocess('1hp')
        wdcnn = WDCNN_Net().cuda()
        optimizer = optim.Adam(wdcnn.parameters(), lr=self.lr)

        for e in range(1, self.epochs+1):
            train_loss, train_acc = self._fit(e, wdcnn, optimizer, [train_data,train_label])
            val_loss, val_acc = self._evaluate(wdcnn, [test_data,test_label])
            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e][train_acc:%.2f%%][val_acc:%.2f%%]"
                % (e, train_loss, val_loss, train_acc*100, val_acc*100))

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
        def _get_random_sample(data):
            random_idx = random.randint(0,len(data)-self.piece_length)
            r_data = np.reshape(data[random_idx:random_idx+self.piece_length],(1,-1))
            r_data = (r_data - np.min(r_data)) / (np.max(r_data) - np.min(r_data))
            return r_data
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in range(self.batches):
            random_idx = np.random.randint(0,len(train_iter[0]),size=self.batch_size)
            pool = ThreadPool()
            data = pool.map(_get_random_sample,[train_iter[0][x] for x in random_idx])
            pool.close()
            pool.join()
            label = [np.reshape(train_iter[1][x],(1,-1)) for x in random_idx]

            data, label = np.concatenate(data,axis=0), np.concatenate(label,axis=0)
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
            # loss = F.l1_loss(output,label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            torch.cuda.empty_cache()        #empty useless variable
        return total_loss / self.batches, correct / total
        
    def _evaluate(self, model, val_iter):
        def _get_random_sample(data):
            random_idx = random.randint(0,len(data)-self.piece_length)
            r_data = np.reshape(data[random_idx:random_idx+self.piece_length],(1,-1))
            r_data = (r_data - np.min(r_data)) / (np.max(r_data) - np.min(r_data))
            return r_data
        
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in range(self.batches):
            random_idx = np.random.randint(0,len(val_iter[0]),size=self.batch_size)
            pool = ThreadPool()
            data = pool.map(_get_random_sample,[val_iter[0][x] for x in random_idx])
            pool.close()
            pool.join()
            label = [np.reshape(val_iter[1][x],(1,-1)) for x in random_idx]

            data, label = np.concatenate(data,axis=0), np.concatenate(label,axis=0)
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
            # loss = F.l1_loss(output,label)
            total_loss += loss.data
            torch.cuda.empty_cache()        #empty useless variable
        return total_loss / self.batches, correct / total


if __name__ == "__main__":
    process = WDCNN()
    process.train()
    