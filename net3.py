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



class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.input2h = nn.Linear(input_size, hidden_size)
        #nn.init.xavier_normal_(self.input2h.weight)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        #nn.init.xavier_normal_(self.h2h.weight)


    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden):
        h_new = torch.tanh(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden

class net3(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)
        #nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
    
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