import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
from torch.autograd import Variable

import numpy as np
import argparse

from generative_RNN import LSTMgen, LSTMclass
from utils import plot_sequence


parser = argparse.ArgumentParser(
    description='Modular Inverse Classification Training With Pytorch')

train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--num_epochs', default=40000,
                    type=int, help='number of epochs used for training')
parser.add_argument('--hidden_size', default=200,
                    type=int, help='number of epochs used for training')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of layers for the LSTM')
parser.add_argument('--train_character', default='a', type=str,
                    help='determines which character you train the generative model on')
parser.add_argument('--noise_in', default=False, type=bool,
                    help='add noise to the first input')
parser.add_argument('--noise_out', default=False, type=bool,
                    help='add noise on the entire target')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--noise_size_input', default=0.05, type=float,
                     help='determines the standard deviation of the noise added to the input')
parser.add_argument('--noise_size_target', default=0.0003, type=float,
                     help='determines the standard deviation of the noise added to the target')
parser.add_argument('--num_chars_per_class', default=1, type=int,
                    help='determines the number of characters that a generative network was trained on')
parser.add_argument('--save_best_only', default=False, type=bool,
                    help='save model with the lowest loss during training')

args = parser.parse_args()

def train(load_file,
          weights_name,
          num_epochs,
          hidden_size,
          output_dim,
          num_layers,
          input_dim,
          show_results=True,
          save_results=True
          ):

    """

    :param load_file:
    :param weights_name:
    :param num_epochs:
    :param hidden_size:
    :param output_dim:
    :param num_layers:
    :param input_dim:
    :param show_results:
    :param save_results:
    :return:
    """

    print("weights name: ", weights_name)
    data = np.load(load_file)

    chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    model = LSTMclass(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers)
    print("parameters: ", model.parameters)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    current_best_loss = float('Inf')

    for epoch in range(num_epochs):

        #randomly choose a character
        rnd = np.random.randint(0, len(chars))
        #get sequence as input
        input_target_pairs = data.item().get(str(chars[rnd]))
        rnd2 = np.random.randint(0, len(input_target_pairs))
        input = torch.Tensor(input_target_pairs[rnd2][1])

        #create one hot as target
        onehot = torch.zeros((1, len(chars)))
        onehot[0][rnd] = 1
        target = torch.Tensor(onehot)
        #print("target size: ", target.size())

        optimizer.zero_grad()
        out = model(input)
        #print("out size: ", out.size())
        loss = loss_function(out, target)
        if (epoch != 0 and epoch % 100 == 0):
            print("epoch: ", epoch, "\tloss: ", loss, 'current best loss: ', current_best_loss)
        loss.backward()
        optimizer.step()

        if args.save_best_only and loss < current_best_loss:
            current_best_loss = loss
            current_best_weights = model.state_dict()

    if not args.save_best_only:
        current_best_weights = model.state_dict()

    torch.save(current_best_weights, 'weights/' + weights_name + '.pth')
    if show_results or save_results:
        with torch.no_grad():
            # test the model
            for c in chars:
                for j in range(len(data.item().get(str(c)))):
                    input_target_pairs = data.item().get(str(c))
                    input = torch.Tensor(input_target_pairs[j][1])
                    target = c
                    #load model and predict
                    model.load_state_dict(current_best_weights)
                    model.hidden = model.init_hidden()
                    pred = model(input).data.numpy()[-1]
                    pred_char = chars[np.argmax(pred)]

                    print("target char: ", target, "predicted char: ", pred_char)


def test(load_file,
         weights_name,
         hidden_size=200,
         input_dim=2,
         output_dim=20,
         num_layers=1):

    #load the data
    data = np.load(load_file)
    #data = data.item()

    #load the model
    model = LSTMclass(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers)
    print("parameters: ", model.parameters)
    print("weights name: ", weights_name)
    checkpoint = torch.load(weights_name)
    model.load_state_dict(checkpoint)

    chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    correct = []
    #test the data
    with torch.no_grad():
        # test the model
        for c in chars:
            for j in range(len(data.item().get(str(c)))):
                input = data.item().get(str(c))
                input = torch.Tensor(input[j])
                print(input.size())
                target = c
                # load model and predict
                model.hidden = model.init_hidden()
                pred = model(input).data.numpy()[-1]
                pred_char = chars[np.argmax(pred)]

                print("target char: ", target, "predicted char: ", pred_char)
                if target == pred_char:
                    correct.append(1)
                else:
                    correct.append(0)

    acc = sum(correct)/len(correct)
    return(acc)




if __name__ == '__main__':

    WEIGHTS_NAME = str('rnn_types_comp_' +
                       'noise_in_{size}_'.format(size=str(args.noise_size_input) if args.noise_in else "0") +
                       'noise_out_{size}_'.format(size=str(args.noise_size_target) if args.noise_out else "0") +
                       'chars_{}'.format(args.num_chars_per_class)
                       )

    """
    train(load_file='../data/sequences_4_handpicked.npy',
         weights_name=WEIGHTS_NAME,
         num_epochs=args.num_epochs,
         hidden_size=args.hidden_size,
         output_dim=20,
         num_layers=args.num_layers,
         input_dim=2,
         show_results=True)
    #"""


    #"""
    load_file_test = '../data/sequences_20_chars_per_class.npy'
    acc = test(load_file=load_file_test,
         weights_name='../RNN/weights/rnn_types_comp_noise_in_0_noise_out_0_chars_1.pth')

    print("accuracy: ", acc)
    #"""


"""
comparison accuracies:
######################
weights_name='../RNN/weights/rnn_types_comp_noise_in_0_noise_out_0_chars_1.pth'
accuracy:  0.455

"""