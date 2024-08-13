from tqdm import tqdm
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import time


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)


    def recurrence(self, input, hidden):
        hx = hidden

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, dim=1)
        h_reset, h_upd, h_new = h_t.chunk(3, dim=1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)

            output = []
            for i in range(input.size(0)):
                hidden = self.recurrence(input[i], hidden)
                output.append(hidden)

            output = torch.stack(output, dim=0)
            return output, hidden

class net5(nn.Module):
    """Recurrent network model."""
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        self.rnn = GRUCell(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity
    

    def train_model(self, dataset, num_epochs, lr, device='cpu'):
        self = self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        running_loss = 0
        running_acc = 0
        start_time = time.time()
        print('Training network...')
        for i in range(num_epochs):
            inputs, labels = dataset()
            inputs = torch.from_numpy(inputs).type(torch.float).to(device)
            labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

            optimizer.zero_grad()
            output, _ = self(inputs)
            #output = output.view(-1, output_size)
            loss = criterion(output.view(-1, self.fc.out_features), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                running_loss /= 100
                print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                    i+1, running_loss, time.time() - start_time))
                running_loss = 0
        self = self.to('cpu')
        return self
