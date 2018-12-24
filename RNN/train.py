import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
from torch.autograd import Variable

import numpy as np
import argparse

from generative_RNN import LSTMgen
from utils import plot_sequence



parser = argparse.ArgumentParser(
    description='Modular Inverse Classification Training With Pytorch')

train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--num_epochs', default=10000,
                    type=int, help='number of epochs used for training')
parser.add_argument('--hidden_size', default=10,
                    type=int, help='number of epochs used for training')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of layers for the LSTM')
parser.add_argument('--train_character', default='a', type=str,
                    help='determines which character you train the generative model on')
parser.add_argument('--noise', default=True, type=bool,
                    help='add noise to the first input and on the entire target')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--weights_name', default='rnn_a_10_noise_out00001_2c', type=str,
                    help='filename for weights')
parser.add_argument('--noise_size_input', default=0, type=float,
                     help='determines the standard deviation of the noise added to the input')
parser.add_argument('--noise_size_target', default=0.0001, type=float,
                     help='determines the standard deviation of the noise added to the target')
parser.add_argument('--save_best_only', default=True, type=bool,
                    help='save model with the lowest loss during training')
#parser.add_argument('--use_deltas', default=False, type=bool,
#                    help='use differences between coordinates or absolute coordinates as target')


args = parser.parse_args()

if __name__ == '__main__':

    sequences = np.load('../data/sequences_2_chars_per_class.npy')
    targets = sequences.item().get(str(args.train_character))
    #plot_sequence(target, swapaxis=True, title="target")
    targets = torch.Tensor(targets)


    # 2 prepare input (output is target)
    zeros = torch.zeros(205)
    zeros[0] = 1
    input_sequence = zeros.view((205, 1))
    print("input_sequence: ", input_sequence.size())

    model = LSTMgen(input_dim = 1, hidden_size=args.hidden_size, output_dim=2, num_layers=args.num_layers)
    #hidden = model.init_hidden()
    print("parameters: ", model.parameters)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    current_best_loss = float('Inf')


    for epoch in range(args.num_epochs):

        # prep data
        input_sequence[0] = 1
        if targets.size(0) > 1:   #randomly choose a target if trained on multiple sequences
            rnd = np.random.randint(0,2)
            target = targets[rnd]
        target = Variable(target, requires_grad=False)
        if args.noise:
            m = normal.Normal(0.0, args.noise_size_input)
            input_sequence[0] += m.sample()
            n = normal.Normal(0.0, args.noise_size_target)
            target += n.sample(sample_shape=(205,2))
        input_sequence = Variable(input_sequence, requires_grad=False)
        #print("input_sequence[0]: ", input_sequence[0])




        optimizer.zero_grad()
        out = model(input_sequence)
        loss = loss_function(out, target)
        if (epoch != 0 and epoch % 100 == 0):
            print("epoch: ", epoch, "\tloss: ", loss.item(), 'current best loss: ', current_best_loss.item())
        loss.backward()
        optimizer.step()

        if args.save_best_only and loss < current_best_loss:
            current_best_loss = loss
            current_best_weights = model.state_dict()

    torch.save(current_best_weights, 'weights/' + args.weights_name + '.pth')
    with torch.no_grad():
        #test the model
        zeros = torch.zeros(len(target))
        zeros[0] = 1
        input_sequence = zeros.view((len(target), 1))
        print("input: ", input_sequence)
        model.load_state_dict(current_best_weights)
        model.hidden = model.init_hidden()
        pred = model(input_sequence).data.numpy()
        print(type(pred))
        print(np.shape(pred))
        print("prediction: ", pred)
        print("target: ", target.data.numpy())
        plot_sequence(pred, swapaxis=True, title="test")
        plot_sequence(target.data.numpy(), swapaxis=True, title="target")