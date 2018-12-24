import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from generative_RNN import LSTMgen
from modular_RNN import ModularRNN
from utils import plot_sequence

if __name__ == '__main__':

    sequences = np.load('../data/sequences.npy')
    sequences_all = np.load('../data/sequences_all.npy')

    train_classes = sequences_all[0]
    train_sequences = sequences_all[1]

    index = 0
    print("train_classes[index]: ", train_classes[index])
    test_target = train_sequences[index]
    plot_sequence(test_target, title='test_target', swapaxis=True)
    test_target = torch.Tensor(test_target)

    # plot_sequence(target, swapaxis=True, title="target")
    target_a = sequences.item().get('a')
    target_b = sequences.item().get('b')
    target_c = sequences.item().get('c')
    target_d = sequences.item().get('d')
    target_e = sequences.item().get('e')
    target_a = torch.Tensor(target_a)
    target_b = torch.Tensor(target_b)
    target_c = torch.Tensor(target_c)
    target_d = torch.Tensor(target_d)
    target_e = torch.Tensor(target_e)


    model_a = LSTMgen(input_dim=1, hidden_size=10, output_dim=2, num_layers=1)
    checkpoint_a = torch.load("weights/rnn_a_10_noise_out00001_2c.pth")
    model_a.load_state_dict(checkpoint_a)
    print("parameters: ", model_a.parameters)

    model_b = LSTMgen(input_dim=1, hidden_size=10, output_dim=2, num_layers=1)
    checkpoint_b = torch.load("weights/rnn_b_10_noise_out00001_2c.pth")
    model_b.load_state_dict(checkpoint_b)

    model_c = LSTMgen(input_dim=1, hidden_size=10, output_dim=2, num_layers=1)
    checkpoint_c = torch.load("weights/rnn_c_10_noise_out00001_2c.pth")
    model_c.load_state_dict(checkpoint_c)

    model_d = LSTMgen(input_dim=1, hidden_size=10, output_dim=2, num_layers=1)
    checkpoint_d = torch.load("weights/rnn_d_10_noise_out00001_2c.pth")
    model_d.load_state_dict(checkpoint_d)

    model_e = LSTMgen(input_dim=1, hidden_size=10, output_dim=2, num_layers=1)
    checkpoint_e = torch.load("weights/rnn_e_10_noise_out00001_2c.pth")
    model_e.load_state_dict(checkpoint_e)

    LoN = [model_a, model_b, model_c, model_d, model_e]

    modularRNN = ModularRNN(list_of_networks = LoN)

    modularRNN.inverse_classification_all(list_of_networks=LoN, target=test_target, iterations=0)