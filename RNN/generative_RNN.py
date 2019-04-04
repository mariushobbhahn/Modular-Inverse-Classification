import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np


class LSTMgen(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim, num_layers):
        super(LSTMgen, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        #define lstm cell
        self.lstm = nn.LSTM(input_dim, hidden_size, self.num_layers)

        #define fully connected
        self.FC =  nn.Linear(hidden_size, output_dim)

        #initialize hidden layer
        #self.hidden = self.init_hidden()


    def init_hidden(self, x = None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if x == None:
            return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        else:
            return (Variable(x[0].data), Variable(x[1].data))


    def forward(self, input):

        self.hidden = self.init_hidden()
        #get lstm outputs
        lstm_out, self.hidden = self.lstm(
            input.view(len(input), 1, -1), self.hidden)

        #print("LSTM_out size: ", lstm_out.size())

        #through fully connected layer:
        #print("input size of FC: ", lstm_out.view(len(input), -1).size())
        coordinates = self.FC(lstm_out.view(len(input), -1))
        return coordinates



class LSTMclass(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim, num_layers):
        super(LSTMclass, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        #define lstm cell
        self.lstm = nn.LSTM(input_dim, hidden_size, self.num_layers)

        #define fully connected
        self.FC =  nn.Linear(hidden_size, output_dim)

        #initialize hidden layer
        #self.hidden = self.init_hidden()


    def init_hidden(self, x = None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if x == None:
            return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        else:
            return (Variable(x[0].data), Variable(x[1].data))


    def forward(self, input):

        self.hidden = self.init_hidden()
        #get lstm outputs
        lstm_out, self.hidden = self.lstm(
            input.view(len(input), 1, -1), self.hidden)

        my_class = self.FC(lstm_out[-1])
        return my_class

