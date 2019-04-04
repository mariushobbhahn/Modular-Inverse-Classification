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
parser.add_argument('--num_epochs', default=40000,
                    type=int, help='number of epochs used for training')
parser.add_argument('--hidden_size', default=10,
                    type=int, help='number of epochs used for training')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of layers for the LSTM')
parser.add_argument('--train_character', default='b', type=str,
                    help='determines which character you train the generative model on')
parser.add_argument('--noise_in', default=False, type=bool,
                    help='add noise to the first input')
parser.add_argument('--noise_out', default=True, type=bool,
                    help='add noise on the entire target')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
#parser.add_argument('--weights_name', default='rnn_z_10_noise_out00001_2c', type=str,
#                    help='filename for weights')
parser.add_argument('--noise_size_input', default=0.05, type=float,
                     help='determines the standard deviation of the noise added to the input')
parser.add_argument('--noise_size_target', default=0.0003, type=float,
                     help='determines the standard deviation of the noise added to the target')
parser.add_argument('--num_chars_per_class', default=1, type=int,
                    help='determines the number of characters that a generative network was trained on')
parser.add_argument('--save_best_only', default=True, type=bool,
                    help='save model with the lowest loss during training')
#parser.add_argument('--use_deltas', default=False, type=bool,
#                    help='use differences between coordinates or absolute coordinates as target')

args = parser.parse_args()



def train_single(load_file,
                 weights_name,
                 character,
                 num_epochs,
                 hidden_size,
                 output_dim,
                 num_layers,
                 input_dim,
                 show_results=True,
                 save_results=True
                 ):
    """
    train a single model

    :return: nothing, model is saved in weights name file
    """

    print("weights name: ", weights_name)
    data = np.load(load_file)
    input_target_pairs = data.item().get(str(character))
    zero_sequence = torch.zeros((205, input_dim))

    model = LSTMgen(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers)
    # hidden = model.init_hidden()
    print("parameters: ", model.parameters)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    current_best_loss = float('Inf')

    for epoch in range(num_epochs):

        # prep data
        #randomly choose a input-target pair if trained on types
        rnd = np.random.randint(0, input_dim)
        input = torch.Tensor(input_target_pairs[rnd][0])
        input_sequence = zero_sequence
        input_sequence[0] = input
        target = torch.Tensor(input_target_pairs[rnd][1])

        #add noise if specified
        if args.noise_in:
            m = normal.Normal(0.0, args.noise_size_input)
            input_sequence[0] += m.sample((input_dim, ))
            input_sequence[0] = input_sequence[0].clamp(min=0, max=1)
        if args.noise_out:
            n = normal.Normal(0.0, args.noise_size_target)
            target += n.sample(sample_shape=(205,2))
        input_sequence = Variable(input_sequence, requires_grad=False)
        #print("input: ", input_sequence[0])

        #optimize
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

    torch.save(current_best_weights, 'weights/' + weights_name + '.pth')
    if show_results or save_results:
        with torch.no_grad():
            # test the model
            zero_sequence = torch.zeros((205, input_dim))
            #print("input: ", input_sequence)
            for i in range(input_dim):
                input = torch.Tensor(input_target_pairs[i][0])
                input_sequence = zero_sequence
                input_sequence[0] = input
                target = torch.Tensor(input_target_pairs[i][1])
                model.load_state_dict(current_best_weights)
                model.hidden = model.init_hidden()
                pred = model(input_sequence).data.numpy()
                #print(type(pred))
                #print(np.shape(pred))
                #print("prediction: ", pred)
                #print("target: ", target.data.numpy())
                if save_results:
                    plot_name = 'figures/' + weights_name + 'type_{}'.format(i) + '.png'
                    print("saving plot of result at: ", plot_name)
                plot_sequence(pred, swapaxis=True, title="test", show=show_results, save=save_results, filename=plot_name)
                plot_sequence(target.data.numpy(), swapaxis=True, show=show_results, save=False, title="target")



def train_multiple(char_list,
                 weights_name,
                 load_file,
                 num_epochs,
                 hidden_size,
                 output_dim,
                 num_layers,
                 input_dim,
                 show_results=True,
                 save_results=True):

    for char in char_list:
        weights_name_f = weights_name.format(char)
        train_single(weights_name=weights_name_f,
                 load_file=load_file,
                 character=char,
                 num_epochs=num_epochs,
                 hidden_size=hidden_size,
                 output_dim=output_dim,
                 num_layers=num_layers,
                 input_dim=input_dim,
                 show_results=show_results,
                 save_results=save_results
                 )

    print("Done training models for: ", char_list)


if __name__ == '__main__':


    WEIGHTS_NAME = str('rnn_types_{}_'  +   #.format(args.train_character) + '_' +
                       'noise_in_{size}_'.format(size=str(args.noise_size_input) if args.noise_in else "0") +
                       'noise_out_{size}_'.format(size=str(args.noise_size_target) if args.noise_out else "0") +
                       'chars_{}'.format(args.num_chars_per_class)
                        )

    """
    train_single(load_file='../data/sequences_4_handpicked.npy',
                 weights_name=WEIGHTS_NAME,
                 character=args.train_character,
                 num_epochs=args.num_epochs,
                 hidden_size=args.hidden_size,
                 output_dim=2,
                 num_layers=args.num_layers,
                 input_dim=4,
                 show_results=True)
    #"""


    #"""
    train_multiple(load_file='../data/sequences_4_handpicked.npy',    #'../data/sequences_2_chars_per_class.npy',
                 char_list=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z'],
                 weights_name=WEIGHTS_NAME,
                 num_epochs=args.num_epochs,
                 hidden_size=args.hidden_size,
                 output_dim=2,
                 num_layers=args.num_layers,
                 input_dim=4,
                 show_results=False,
                 save_results=True
                 )
    #"""

