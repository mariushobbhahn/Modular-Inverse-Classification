import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from generative_RNN import LSTMgen
from modular_RNN import ModularRNN
#from Inverse_classification.inverse_classification import get_list_of_models #fucking imports in python
from utils import plot_sequence

def get_list_of_models(model_name, chars, input_dim=1, hidden_size=10, output_dim=2, num_layers=1):
    """
    returns a list of trained models where all have the same weight name
    except for the specified character.

    :param model_name: string with {} at the place where the letter is to be inserted
    :param chars: list of characters for which the model should be loaded
    :return: list of loaded models
    """

    LoM = []
    for c in chars:
        weights_name = model_name.format(c)
        model = LSTMgen(input_dim=input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers)
        checkpoint = torch.load(weights_name)
        model.load_state_dict(checkpoint)
        LoM.append(model)

    return (LoM)


if __name__ == '__main__':


    load_file = '../data/sequences_4_handpicked.npy'
    #load_file = '../data/sequences_all.npy'
    model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0_chars_1.pth'

    data = np.load(load_file)
    #classes, sequences = data[0], data[1]
    #print(np.shape(classes), np.shape(sequences))
    character = 'b'
    input_target_pairs = data.item().get(str(character))
    #zero_sequence = torch.zeros((205, 4))

    model = LSTMgen(input_dim=4, hidden_size=10, output_dim=2, num_layers=1)
    #checkpoint = torch.load(model_name)
    #model.load_state_dict(checkpoint)

    LoM = get_list_of_models(model_name=model_name,
                             chars=['b'],
                             input_dim=4
                             )

    #number = 3      #denotes the number of the types: int from 0 to 3
    #input = torch.Tensor(input_target_pairs[number][0])
    #input_sequence = zero_sequence
    #input_sequence[0] = input


    for number in range(4):
        target = torch.Tensor(input_target_pairs[number][1])

    #for j in range(200):
    #    target = torch.Tensor(sequences[i])

        modRNN = ModularRNN(list_of_networks=LoM)

        modRNN.inverse_classification_rnn(gen_net=LoM[0], target=target ,iterations=100)




