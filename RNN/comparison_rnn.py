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
from data.comparison_class import ComparisonData
import torch.utils.data as data


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
parser.add_argument('--batch_size', default=16, type=int, help='batch size for training')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--noise_size_input', default=0.05, type=float,
                     help='determines the standard deviation of the noise added to the input')
parser.add_argument('--noise_size_target', default=0.0003, type=float,
                     help='determines the standard deviation of the noise added to the target')
parser.add_argument('--num_chars_per_class', default=1, type=int,
                    help='determines the number of characters that a generative network was trained on')
parser.add_argument('--save_best_only', default=True, type=bool,
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
          save_results=True,
          many_samples=True,
          batch_size=16
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
    load_file = '../data/sequences_comparison_class.npy'
    my_data = np.load(load_file)
    train_classes, train_sequences, val_classes, val_sequences, test_classes, test_sequences = my_data

    train_dataset = ComparisonData(classes=train_classes, sequences=train_sequences)
    val_dataset = ComparisonData(classes=val_classes, sequences=val_sequences)

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  pin_memory=True)

    val_loader = data.DataLoader(val_dataset, args.batch_size,
                                   num_workers=4,
                                   shuffle=True,
                                   pin_memory=True)

    model = nn.DataParallel(LSTMclass(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers), dim=1).cuda()
    #model = LSTMclass(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers).cuda()
    print("parameters: ", model.parameters)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    current_best_val_loss = float('Inf')

    for epoch in range(num_epochs):

        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

            optimizer.zero_grad()
            out = model(local_batch, batch_size=len(local_batch))
            batch_loss = loss_function(out, local_labels)
            batch_loss.backward()
            optimizer.step()

        for local_batch, local_labels in val_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

            optimizer.zero_grad()
            out = model(local_batch, batch_size=len(local_batch))
            val_loss = loss_function(out, local_labels)

        if args.save_best_only and val_loss < current_best_val_loss:
            current_best_val_loss = val_loss
            current_best_weights = model.state_dict()

        if epoch != 0 and epoch % 100 == 0:
            print("epoch: ", epoch, "\tbatch_loss: ", batch_loss, 'val_loss: ', val_loss, 'current best loss: ',  current_best_val_loss)


    if not args.save_best_only:
        current_best_weights = model.state_dict()

    torch.save(current_best_weights, 'weights/' + weights_name + '.pth')
    if show_results or save_results:
        with torch.no_grad():
            if many_samples:
                for i in range(len(train_classes)):
                    input = torch.Tensor(train_targets[i])
                    target = train_classes[i]

                    model.load_state_dict(current_best_weights)
                    model.hidden = model.init_hidden()
                    pred = model(input).data.numpy()[-1]
                    pred_char = chars[np.argmax(pred)]

                    print("target char: ", target, "predicted char: ", pred_char)

            else:
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
         num_layers=1,
         many_samples=False):

    #load the data
    data = np.load(load_file)
    #data = data.item()
    if many_samples:
        _, _, _, _, test_classes, test_targets = data

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
        if many_samples:
            for i in range(len(test_classes)):
                input = torch.Tensor(test_targets[i])
                target = test_classes[i]
                model.hidden = model.init_hidden()
                pred = model(input).data.numpy()[-1]
                pred_char = chars[np.argmax(pred)]

                print("target char: ", target, "predicted char: ", pred_char)
                if target == pred_char:
                    correct.append(1)
                else:
                    correct.append(0)

        else:
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

    WEIGHTS_NAME = str('rnn_types_comp_many_samples_' +
                       'noise_in_{size}_'.format(size=str(args.noise_size_input) if args.noise_in else "0") +
                       'noise_out_{size}_'.format(size=str(args.noise_size_target) if args.noise_out else "0")
                       #'chars_{}'.format(args.num_chars_per_class)
                       )

    #"""
    train(load_file='../data/sequences_comparison_class.npy',
         weights_name=WEIGHTS_NAME,
         num_epochs=args.num_epochs,
         hidden_size=args.hidden_size,
         output_dim=20,
         num_layers=args.num_layers,
         input_dim=2,
         show_results=True,
         many_samples=True)
    #"""


    """
    load_file_test = '../data/sequences_comparison_class.npy'
    acc = test(load_file=load_file_test,
         weights_name='../RNN/weights/rnn_types_comp_many_samplesnoise_in_0_noise_out_0_chars_1.pth',
               many_samples=True)

    print("accuracy: ", acc)
    #"""


"""
comparison accuracies:
######################
weights_name='../RNN/weights/rnn_types_comp_noise_in_0_noise_out_0_chars_1.pth'
accuracy:  0.455

"""