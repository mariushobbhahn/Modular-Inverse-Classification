import torch
import numpy as np

from RNN.generative_RNN import LSTMgen
from RNN.modular_RNN import ModularRNN



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

def inversely_classify_all_sequences(sequences,
                                     classes,
                                     modular_RNN,
                                     iterations_per_ic,
                                     print_loss=True,
                                     print_accuracy=True,
                                     each_character_ind=True
                                     ):


    """

    :param sequences:  target_sequences to be classified
    :param print_loss:  print MSE_loss between prediction and target
    :param print_accuracy:  print accuracy of prediction
    :param each_character_ind: show all metrics for each character individually.

    :return: a tuple of loss, accuracy for all characters and combined
    However this is printed during the process as well
    """

    list_of_chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    loss_per_char = [[] for i in range(len(np.unique(classes)))]
    correct_per_char = [[] for i in range(len(np.unique(classes)))]

    correct = []
    losses = []

    for i in range(len(sequences)):

        print("index: ", i)

        target = sequences[i]
        target = torch.FloatTensor(target)

        loss_list = modular_RNN.inverse_classification_all(target=target, iterations=iterations_per_ic)
        loss_min = np.min(loss_list)
        prediction_idx = np.argmin(loss_list)
        prediction = list_of_chars[prediction_idx]
        print("prediction: ", prediction)
        print("real: ", classes[i])

        #append losses to the individual characterlists
        if each_character_ind:
            target_idx = list_of_chars.index(classes[i])
            loss_per_char[target_idx].append(loss_min)
            if classes[i] == prediction:
                correct_per_char[target_idx].append(1)
            else:
                correct_per_char[target_idx].append(0)


        #append losses overall
        losses.append(loss_min)
        if classes[i] == prediction:
            correct.append(1)
        else:
            correct.append(0)


    n = len(classes)
    loss = sum(losses)/n
    accuracy = sum(correct)/n
    if print_loss:
        print("overall loss: ", loss)
        if each_character_ind:
            for i in range(len(loss_per_char)):
                print("loss for character: ", list_of_chars[i], sum(loss_per_char[i])/len(loss_per_char[i]))
    if print_accuracy:
        print("overall accuracy: ", accuracy)
        if each_character_ind:
            for i in range(len(correct_per_char)):
                print("accuracy for character: ", list_of_chars[i], sum(correct_per_char[i]) / len(correct_per_char[i]))

    return(loss, accuracy)





if __name__ == '__main__':

    sequences_all = np.load('../data/sequences_all.npy')

    #model_name = '../RNN/weights/rnn_{}_10_noise_out00001_2c.pth'
    model_name = '../RNN/weights/rnn_types_{}_noise_in_0.2_noise_out_0_chars_1.pth'
    LoM = get_list_of_models(model_name=model_name,
                             chars=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z'],
                             input_dim=4
                             )

    test_classes = sequences_all[0]
    test_sequences = sequences_all[1]

    idx = np.random.choice(np.arange(len(test_classes)), 200, replace=False)
    test_sequences_random = test_sequences[idx]
    test_classes_random = test_classes[idx]

    modRNN = ModularRNN(list_of_networks=LoM)

    _,_ = inversely_classify_all_sequences(sequences=test_sequences_random,
                                           classes=test_classes_random,
                                           modular_RNN=modRNN,
                                           iterations_per_ic=50
                                           )



"""

#rnn_{}_noise_in_0_noise_out_0_chars_1.pth', iterations_per_ic=0

overall loss:  0.1364216336733807
loss for character:  a 0.07021037873346359
loss for character:  b 0.07821024092845619
loss for character:  c 0.07291616800312813
loss for character:  d 0.17424756164352098
loss for character:  e 0.06069928083282624
loss for character:  g 0.2733142889208264
loss for character:  h 0.10692455135285854
loss for character:  l 0.08513871906325221
loss for character:  m 0.31845334817965826
loss for character:  n 0.0884380762775739
loss for character:  o 0.1965368628501892
loss for character:  p 0.04862734992057085
loss for character:  q 0.09187156340340152
loss for character:  r 0.04510198533535004
loss for character:  s 0.05648152243035535
loss for character:  u 0.054800479171367794
loss for character:  v 0.08634670058058368
loss for character:  w 0.12131315576178688
loss for character:  y 0.3071092309223281
loss for character:  z 0.20609313187499842
overall accuracy:  0.775
accuracy for character:  a 1.0
accuracy for character:  b 1.0
accuracy for character:  c 0.6923076923076923
accuracy for character:  d 1.0
accuracy for character:  e 1.0
accuracy for character:  g 0.4444444444444444
accuracy for character:  h 0.7
accuracy for character:  l 0.625
accuracy for character:  m 0.7333333333333333
accuracy for character:  n 0.6666666666666666
accuracy for character:  o 0.2
accuracy for character:  p 1.0
accuracy for character:  q 0.875
accuracy for character:  r 1.0
accuracy for character:  s 1.0
accuracy for character:  u 0.9230769230769231
accuracy for character:  v 0.5555555555555556
accuracy for character:  w 0.42857142857142855
accuracy for character:  y 0.8888888888888888
accuracy for character:  z 0.7222222222222222
"""

"""

model_name = '../RNN/weights/rnn_types_{}_noise_in_0.1_noise_out_0_chars_1.pth'

overall loss:  0.09452036403345118
loss for character:  a 0.039771830753630236
loss for character:  b 0.028846671293851815
loss for character:  c 0.050183663098141554
loss for character:  d 0.05708870374863701
loss for character:  e 0.0632401327136904
loss for character:  g 0.14576795417815447
loss for character:  h 0.05416293192485517
loss for character:  l 0.06061251829669345
loss for character:  m 0.20974740386009216
loss for character:  n 0.07756220282317372
loss for character:  o 0.10067061486188322
loss for character:  p 0.07649109613460799
loss for character:  q 0.2074287554456128
loss for character:  r 0.052675666908423104
loss for character:  s 0.036874350012195384
loss for character:  u 0.05724466759711504
loss for character:  v 0.1097097190347715
loss for character:  w 0.11060300448702441
loss for character:  y 0.1538952199875244
loss for character:  z 0.22998151422611304
overall accuracy:  0.885
accuracy for character:  a 0.8888888888888888
accuracy for character:  b 1.0
accuracy for character:  c 1.0
accuracy for character:  d 1.0
accuracy for character:  e 0.9
accuracy for character:  g 1.0
accuracy for character:  h 0.9090909090909091
accuracy for character:  l 0.9090909090909091
accuracy for character:  m 1.0
accuracy for character:  n 0.875    
accuracy for character:  o 0.625
accuracy for character:  p 0.8333333333333334
accuracy for character:  q 0.8888888888888888
accuracy for character:  r 1.0
accuracy for character:  s 1.0
accuracy for character:  u 1.0
accuracy for character:  v 0.6153846153846154
accuracy for character:  w 0.6666666666666666
accuracy for character:  y 0.9285714285714286
accuracy for character:  z 0.7857142857142857
"""

"""
model_name = '../RNN/weights/rnn_types_{}_noise_in_0.2_noise_out_0_chars_1.pth'

overall loss:  0.1009666990597907
loss for character:  a 0.04940825385543016
loss for character:  b 0.047519674317704305
loss for character:  c 0.056827050323287644
loss for character:  d 0.09314793888479471
loss for character:  e 0.046022218150588184
loss for character:  g 0.09932621195912361
loss for character:  h 0.10568072157911956
loss for character:  l 0.0549586463926567
loss for character:  m 0.08069473792177935
loss for character:  n 0.08550719047586124
loss for character:  o 0.06646208489213937
loss for character:  p 0.044514232522083655
loss for character:  q 0.12276475727558137
loss for character:  r 0.03294102359879097
loss for character:  s 0.02584048085069905
loss for character:  u 0.055799058431552515
loss for character:  v 0.1311474971783658
loss for character:  w 0.13088178945084414
loss for character:  y 0.35517056773488337
loss for character:  z 0.24379178324791914
overall accuracy:  0.855
accuracy for character:  a 0.8461538461538461
accuracy for character:  b 1.0
accuracy for character:  c 1.0
accuracy for character:  d 1.0
accuracy for character:  e 1.0
accuracy for character:  g 0.625
accuracy for character:  h 0.75
accuracy for character:  l 0.8888888888888888
accuracy for character:  m 1.0
accuracy for character:  n 0.5
accuracy for character:  o 0.8571428571428571
accuracy for character:  p 1.0
accuracy for character:  q 0.8
accuracy for character:  r 1.0
accuracy for character:  s 1.0
accuracy for character:  u 0.8888888888888888
accuracy for character:  v 0.4166666666666667
accuracy for character:  w 1.0
accuracy for character:  y 0.7692307692307693
accuracy for character:  z 0.6666666666666666
"""
