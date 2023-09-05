import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import relu
import numpy as np


class DynamicalSystemDataset(Dataset):

    def __init__(self, data_file, inp_len, out_len):
        super(DynamicalSystemDataset, self).__init__()
        self.data = np.loadtxt(data_file)
        self.data_mean = torch.Tensor([self.data.mean()])
        self.data_std = torch.Tensor([self.data.std()])
        self.inp_len = inp_len
        self.out_len = out_len
        self.n1 = self.data.shape[0]
        self.n2 = self.data.shape[1] - self.inp_len - self.out_len + 1
        self.dim = self.n1*self.n2

    def __len__(self):
        return self.dim

    def __getitem__(self, index):
        i = index//self.n2
        j = index % self.n2
        src = torch.Tensor(self.data[i, j:j+self.inp_len]).unsqueeze(-1)
        trg = torch.Tensor(self.data[i, j+self.inp_len:j+self.inp_len+self.out_len])
        return {'src': src,
                'trg': trg}


class SimpleLSTM(nn.Module):

    def __init__(self, inp_size=1, hid_size=64, out_size=2, layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm_layer = torch.nn.LSTM(inp_size, hid_size, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hid_size, out_size)

    def forward(self, x):
        x, (_, _) = self.lstm_layer(x)
        y = x[:, -1, :]
        x = self.fc(y)
        y = x[:, 1].unsqueeze(1)
        # y = relu(y)
        # y = y**2
        y = torch.exp(y)
        return torch.cat((x[:, 0].unsqueeze(1), y), dim=1)


class SimpleLinear(nn.Module):

    def __init__(self, inp_size=3, hid_size=64, out_size=2, num_layers=2):
        super(SimpleLinear, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(inp_size, hid_size))
        for i in range(self.num_layers - 2):
            self.linear.append(torch.nn.Linear(hid_size, hid_size))
        self.linear.append(torch.nn.Linear(hid_size, out_size))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear[i](x)
            x = relu(x)
            # x = torch.sin(x)
        x = self.linear[-1](x)
        y = x[:, 1].unsqueeze(1)
        # y = relu(y)
        # y = y**2
        y = torch.exp(y)
        return torch.cat((x[:, 0].unsqueeze(1), y), dim=1)
