import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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

def evaluate_inverse_classification(data,
                                    model_name,
                                    list_of_characters=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
                                    ):
    """

    :param data:
    :param model_name: i.e. '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0_chars_1.pth'
    :param list_of_characters: list of character models to be tested
    :return: a list of results of inverse classification and correct one hot-vectors
    """
    #model = LSTMgen(input_dim=4, hidden_size=10, output_dim=2, num_layers=1)
    # checkpoint = torch.load(model_name)
    # model.load_state_dict(checkpoint)

    LoM = get_list_of_models(model_name=model_name,
                             chars=list_of_characters,
                             input_dim=4
                             )

    pred_true_class_pairs = []
    for idx, char in enumerate(list_of_characters):
        input_target_pairs = data.item().get(str(char))

        for number in range(4):
            target = torch.Tensor(input_target_pairs[number][1])
            true_class = torch.Tensor(input_target_pairs[number][0])

            modRNN = ModularRNN(list_of_networks=LoM)

            filename = 'figures/eval_IC_{}_{}.png'.format(char, number)
            loss, pred_class = modRNN.inverse_classification_rnn(gen_net=LoM[idx], target=target, iterations=50, save_plot=True, show_plot=False, filename=filename, verbose=False)
            print("loss: ", loss, "pred_class: ", pred_class, "true_class: ", true_class)
            pred_true_class_pairs.append([pred_class, true_class])

    return(pred_true_class_pairs)


def get_error(true_pred_class_pairs):
    """

    :param true_pred_class_pairs: list of true_pred_class_pairs
    :return: accuracy and mse
    """
    correct = []
    error = []

    for pair in true_pred_class_pairs:
        #create a one-hot of the maximum value in the prediction:
        print("pair: ", pair)
        pred = pair[0]
        true = pair[1].numpy()
        one_hot_vec = np.zeros_like(pred)
        one_hot_vec[np.argmax(pred)] = 1
        #add error measures to lists
        print("rmse: ", np.sqrt(np.mean(np.square(pred - true))))
        error.append(np.sqrt(np.mean(np.square(pred - true))))
        print("correct: ", np.array_equal(pred, true))
        if np.array_equal(one_hot_vec, true):
            correct.append(1)
        else:
            correct.append(0)

    rmse = np.mean(error)
    accuracy = np.mean(correct)

    return(accuracy, rmse)

def evaluate_IC_unseen_test(data,
                            model_name,
                            list_of_characters=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
                            ):

    #classes, sequences = data[0], data[1]
    #print(np.shape(classes), np.shape(sequences))

    #n = len(classes)
    data = data.item()
    n = len(data[str(list_of_characters[0])])
    print("number of examples per class: ", n)

    LoM = get_list_of_models(model_name=model_name,
                             chars=list_of_characters,
                             input_dim=4
                             )

    losses = []

    for c in list_of_characters:
        target_chars = data[c]
        for i in range(n):
            this_class = c
            this_target = torch.Tensor(target_chars[i])

            modRNN = ModularRNN(list_of_networks=LoM)

            filename = 'figures/eval_IC_unseen_{}_{}.png'.format(this_class, i)
            lom_idx = list_of_characters.index(this_class)
            loss, pred_class = modRNN.inverse_classification_rnn(gen_net=LoM[lom_idx], target=this_target, iterations=0,
                                                                 save_plot=False, show_plot=False, filename=filename,
                                                                 verbose=False)
            print("loss: ", loss, "pred_class: ", pred_class)

            losses.append(loss)

    return(losses)

def create_cross_classification(data,
                                model_name,
                                list_of_characters=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z'],
                                filename='loss_cross.npy'):
    """
    creates an n by n array of the average losses that a model for char i has for classifying the chars of char j


    :return: an array of the average losses for all characters
    """

    LoM = get_list_of_models(model_name=model_name,
                             chars=list_of_characters,
                             input_dim=4
                             )

    modRNN = ModularRNN(list_of_networks=LoM)

    losses_all = []

    for i, char_x in enumerate(list_of_characters):

        losses_class = []

        for j, char_y in enumerate(list_of_characters):
            input_target_pairs = data.item().get(str(char_y))
            losses_char = []

            for number in range(4):
                target = torch.Tensor(input_target_pairs[number][1])
                #true_class = torch.Tensor(input_target_pairs[number][0])


                pic_filename = 'figures/cross_class_{}_{}_{}.png'.format(char_x, char_y, number)
                loss, pred_class = modRNN.inverse_classification_rnn(gen_net=LoM[i], target=target, iterations=50,
                                                                     save_plot=False, show_plot=False, filename=pic_filename,
                                                                     verbose=False)
                print("current gen net: ", char_x, "current target char: ", char_y, "loss: ", loss)
                losses_char.append(loss)

            losses_class.append(np.mean(losses_char))

        losses_all.append(losses_class)

    losses_all = np.array(losses_all)
    print("saving file at: ", filename)
    np.save(filename, losses_all)
    return(losses_all)

def heat_map_for_cross_table(filename,
                             list_of_characters=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z'],
                             ):

    data = np.load(filename)
    plt.imshow(data, cmap='pink', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(list_of_characters)), list_of_characters)
    plt.yticks(np.arange(len(list_of_characters)), list_of_characters)
    plt.xlabel('average loss of generated char')
    plt.ylabel('test target')
    plt.title('heat map of the loss')
    plt.show()


if __name__ == '__main__':


    load_file = '../data/sequences_4_handpicked.npy'
    load_file_test = '../data/sequences_20_chars_per_class.npy'
    model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0.0001_chars_1.pth'

    data = np.load(load_file)
    data_test = np.load(load_file_test)

    #number = 3      #denotes the number of the types: int from 0 to 3
    #input = torch.Tensor(input_target_pairs[number][0])
    #input_sequence = zero_sequence
    #input_sequence[0] = input


    """
    pred_true_class_pairs =  evaluate_inverse_classification(data=data, model_name=model_name)


    acc, rmse = get_error(pred_true_class_pairs)

    print("acc: ", acc)
    print("error: ", rmse)
    #"""

    #"""
    losses = evaluate_IC_unseen_test(data=data_test,
                                     model_name=model_name)

    print('average loss: ', np.mean(losses), 'losses: ', losses)
    #"""

    """
    loss_cross = create_cross_classification(data=data,
                                             model_name=model_name,
                                             filename='loss_cross_in_005.npy')

    print("loss_cross: ", loss_cross)
    #"""


    """
    heat_map_for_cross_table('loss_cross_in_02.npy')

    #"""


"""
evaluate inverse classification:
################################

model_name = '../RNN/weights/rnn_types_{}_noise_in_0_noise_out_0_chars_1.pth'
acc:  0.55
error:  0.30108267

model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0_chars_1.pth'
acc:  0.85
error:  0.18532161

model_name = '../RNN/weights/rnn_types_{}_noise_in_0.2_noise_out_0_chars_1.pth'
acc:  0.85
error:  0.19833142

model_name = '../RNN/weights/rnn_types_{}_noise_in_0_noise_out_0.0001_chars_1.pth'
acc:  0.55
error:  0.3286135

model_name = '../RNN/weights/rnn_types_{}_noise_in_0_noise_out_0.0002_chars_1.pth'
acc:  0.55
error:  0.31428707

model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0.0001_chars_1.pth'
acc:  0.825
error:  0.20416602



"""


"""
previously unseen data average loss:
####################################

model_name = '../RNN/weights/rnn_types_{}_noise_in_0_noise_out_0.0002_chars_1.pth'
50 iterations:   average loss:  0.24183661
0 iterations: average loss:  0.3424529


model_name = '../RNN/weights/rnn_types_{}_noise_in_0_noise_out_0.0001_chars_1.pth'
50 iterations: average loss:  0.25192034
0 iterations: average loss:  0.36481312


model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0_chars_1.pth'
50 iterations: average loss:  0.17815898
0 iterations: average loss:  0.3320325


model_name = '../RNN/weights/rnn_types_{}_noise_in_0.05_noise_out_0_chars_1.pth'
50 iterations: average loss:  0.20017742
0 iterations: average loss:  0.32538423


model_name = '../RNN/weights/rnn_types_{}_noise_in_0.05_noise_out_0_chars_1.pth'
50 iterations: average loss:  0.20088354
0 iterations: average loss:  0.3254273

model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0.0001_chars_1.pth'
50 iterations: average loss:  0.1954365
0 iterations: average loss:  0.3598676

"""
