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


class net2(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, dt,  gain = .85):
    super(net2, self).__init__()
    params = self.initialize_parameters(input_size, hidden_size, output_size, gain)
    self.h0, self.wI, self.wR, self.wO, self.bR, self.bO = params
    self.dt = dt
    tau = 100
    self.alpha = dt / tau

  def initialize_parameters(self, input_size, hidden_size, output_size, gain):
    hscale = 0.1
    ifactor = gain / torch.sqrt(torch.tensor(input_size, dtype=torch.float))
    hfactor = gain / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))
    pfactor = gain / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))
    h0 = nn.Parameter(torch.randn(hidden_size) * hscale)
    wI = nn.Parameter(torch.randn(hidden_size, input_size) * ifactor)
    wR = nn.Parameter(torch.randn(hidden_size, hidden_size) * hfactor)
    wO = nn.Parameter(torch.randn(output_size, hidden_size) * pfactor)
    bR = nn.Parameter(torch.zeros(hidden_size))
    bO = nn.Parameter(torch.zeros(output_size))

    return h0, wI, wR, wO, bR, bO

  def forward(self, x):

    h = self.h0.unsqueeze(0).expand(x.size(1), -1) # Expanding h0 to the batch size
    output = []
    hidden_state = []

    # Process the input sequence one time step at a time
    for t in range(x.size(0)):
        h = torch.tanh(x[t,:,:] @ self.wI.T + self.bR + torch.matmul(h, self.wR))
        o_t = torch.matmul(h, self.wO.T) + self.bO
        output.append(o_t)
        hidden_state.append(h)
        # Stack outputs to have shape [seq_len, batch_size, output_size]
        outputs = torch.stack(output, dim=0)
        hidden_states = torch.stack(hidden_state, dim=0)
    return outputs, hidden_states
  
  def train_model(self, dataset, num_epochs, lr):
      optimizer = optim.Adam(self.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss()

      running_loss = 0
      running_acc = 0
      start_time = time.time()
      print('Training network...')
      for i in range(num_epochs):
          inputs, labels = dataset()
          inputs = torch.from_numpy(inputs).type(torch.float)
          labels = torch.from_numpy(labels.flatten()).type(torch.long)

          optimizer.zero_grad()
          output, _ = self(inputs)
          #output = output.view(-1, output_size)
          loss = criterion(output.view(-1, self.bO.size(0)), labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
          if i % 100 == 99:
              running_loss /= 100
              print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                  i+1, running_loss, time.time() - start_time))
              running_loss = 0
      return self
  
