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

    model_name = '../RNN/weights/rnn_{}_10_noise_out00001_2c.pth'
    #model_name = '../RNN/weights/rnn_{}_noise_in_0_noise_out_0_chars_1.pth'
    LoM = get_list_of_models(model_name=model_name,
                             chars=['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
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
                                           iterations_per_ic=5
                                           )




"""

rnn_{}_noise_in_0_noise_out_0_chars_1.pth', iterations_per_ic=10

overall loss:  tensor(0.4439, grad_fn=<DivBackward0>)
loss for character:  a tensor(0.2376, grad_fn=<DivBackward0>)
loss for character:  b tensor(0.0621, grad_fn=<DivBackward0>)
loss for character:  c tensor(0.0543, grad_fn=<DivBackward0>)
loss for character:  d tensor(0.9318, grad_fn=<DivBackward0>)
loss for character:  e tensor(0.4055, grad_fn=<DivBackward0>)
loss for character:  g tensor(0.7436, grad_fn=<DivBackward0>)
loss for character:  h tensor(0.4270, grad_fn=<DivBackward0>)
loss for character:  l tensor(0.3045, grad_fn=<DivBackward0>)
loss for character:  m tensor(0.8862, grad_fn=<DivBackward0>)
loss for character:  n tensor(0.5312, grad_fn=<DivBackward0>)
loss for character:  o tensor(0.1421, grad_fn=<DivBackward0>)
loss for character:  p tensor(0.1976, grad_fn=<DivBackward0>)
loss for character:  q tensor(0.0884, grad_fn=<DivBackward0>)
loss for character:  r tensor(0.3062, grad_fn=<DivBackward0>)
loss for character:  s tensor(0.4040, grad_fn=<DivBackward0>)
loss for character:  u tensor(0.4884, grad_fn=<DivBackward0>)
loss for character:  v tensor(0.3033, grad_fn=<DivBackward0>)
loss for character:  w tensor(0.4670, grad_fn=<DivBackward0>)
loss for character:  y tensor(1.3835, grad_fn=<DivBackward0>)
loss for character:  z tensor(0.5427, grad_fn=<DivBackward0>)
overall accuracy:  0.425
accuracy for character:  a 0.07692307692307693
accuracy for character:  b 1.0
accuracy for character:  c 1.0
accuracy for character:  d 0.1
accuracy for character:  e 0.0
accuracy for character:  g 0.0
accuracy for character:  h 0.625
accuracy for character:  l 0.09090909090909091
accuracy for character:  m 0.125
accuracy for character:  n 0.0
accuracy for character:  o 0.8181818181818182
accuracy for character:  p 1.0
accuracy for character:  q 1.0
accuracy for character:  r 0.5714285714285714
accuracy for character:  s 1.0
accuracy for character:  u 0.0
accuracy for character:  v 0.36363636363636365
accuracy for character:  w 0.09090909090909091
accuracy for character:  y 0.375
accuracy for character:  z 0.8571428571428571

"""

"""

#rnn_{}_noise_in_0_noise_out_0_chars_1.pth', iterations_per_ic=1

overall loss:  tensor(0.3485, grad_fn=<DivBackward0>)
loss for character:  a tensor(0.0311, grad_fn=<DivBackward0>)
loss for character:  b tensor(0.2382, grad_fn=<DivBackward0>)
loss for character:  c tensor(0.0346, grad_fn=<DivBackward0>)
loss for character:  d tensor(0.5731, grad_fn=<DivBackward0>)
loss for character:  e tensor(0.2910, grad_fn=<DivBackward0>)
loss for character:  g tensor(0.4403, grad_fn=<DivBackward0>)
loss for character:  h tensor(0.2018, grad_fn=<DivBackward0>)
loss for character:  l tensor(0.3073, grad_fn=<DivBackward0>)
loss for character:  m tensor(0.6828, grad_fn=<DivBackward0>)
loss for character:  n tensor(0.4689, grad_fn=<DivBackward0>)
loss for character:  o tensor(0.1433, grad_fn=<DivBackward0>)
loss for character:  p tensor(0.1266, grad_fn=<DivBackward0>)
loss for character:  q tensor(0.1639, grad_fn=<DivBackward0>)
loss for character:  r tensor(0.0979, grad_fn=<DivBackward0>)
loss for character:  s tensor(0.2075, grad_fn=<DivBackward0>)
loss for character:  u tensor(0.3152, grad_fn=<DivBackward0>)
loss for character:  v tensor(0.2290, grad_fn=<DivBackward0>)
loss for character:  w tensor(0.2797, grad_fn=<DivBackward0>)
loss for character:  y tensor(1.0965, grad_fn=<DivBackward0>)
loss for character:  z tensor(0.4146, grad_fn=<DivBackward0>)
overall accuracy:  0.625
accuracy for character:  a 1.0
accuracy for character:  b 0.8333333333333334
accuracy for character:  c 1.0
accuracy for character:  d 0.3076923076923077
accuracy for character:  e 0.7083333333333334
accuracy for character:  g 0.7142857142857143
accuracy for character:  h 0.8333333333333334
accuracy for character:  l 0.7777777777777778
accuracy for character:  m 0.2
accuracy for character:  n 0.0
accuracy for character:  o 0.6923076923076923
accuracy for character:  p 0.75
accuracy for character:  q 0.9090909090909091
accuracy for character:  r 1.0
accuracy for character:  s 1.0
accuracy for character:  u 0.06666666666666667
accuracy for character:  v 0.8
accuracy for character:  w 0.8
accuracy for character:  y 0.2222222222222222
accuracy for character:  z 0.7777777777777778
"""


"""

#rnn_{}_noise_in_0_noise_out_0_chars_1.pth', iterations_per_ic=0

overall loss:  tensor(0.1232, grad_fn=<DivBackward0>)
loss for character:  a tensor(0.0411, grad_fn=<DivBackward0>)
loss for character:  b tensor(0.0519, grad_fn=<DivBackward0>)
loss for character:  c tensor(0.0550, grad_fn=<DivBackward0>)
loss for character:  d tensor(0.1153, grad_fn=<DivBackward0>)
loss for character:  e tensor(0.0975, grad_fn=<DivBackward0>)
loss for character:  g tensor(0.1415, grad_fn=<DivBackward0>)
loss for character:  h tensor(0.1391, grad_fn=<DivBackward0>)
loss for character:  l tensor(0.1277, grad_fn=<DivBackward0>)
loss for character:  m tensor(0.1899, grad_fn=<DivBackward0>)
loss for character:  n tensor(0.1690, grad_fn=<DivBackward0>)
loss for character:  o tensor(0.1578, grad_fn=<DivBackward0>)
loss for character:  p tensor(0.0599, grad_fn=<DivBackward0>)
loss for character:  q tensor(0.1430, grad_fn=<DivBackward0>)
loss for character:  r tensor(0.0744, grad_fn=<DivBackward0>)
loss for character:  s tensor(0.0507, grad_fn=<DivBackward0>)
loss for character:  u tensor(0.0677, grad_fn=<DivBackward0>)
loss for character:  v tensor(0.1329, grad_fn=<DivBackward0>)
loss for character:  w tensor(0.1643, grad_fn=<DivBackward0>)
loss for character:  y tensor(0.2689, grad_fn=<DivBackward0>)
loss for character:  z tensor(0.2232, grad_fn=<DivBackward0>)
overall accuracy:  0.8792439621981099
accuracy for character:  a 0.9941520467836257
accuracy for character:  b 0.9929078014184397
accuracy for character:  c 0.9225352112676056
accuracy for character:  d 0.9554140127388535
accuracy for character:  e 0.9731182795698925
accuracy for character:  g 0.8695652173913043
accuracy for character:  h 0.8818897637795275
accuracy for character:  l 0.7011494252873564
accuracy for character:  m 0.856
accuracy for character:  n 0.6461538461538462
accuracy for character:  o 0.6453900709219859
accuracy for character:  p 0.9618320610687023
accuracy for character:  q 0.9193548387096774
accuracy for character:  r 0.8559322033898306
accuracy for character:  s 0.9774436090225563
accuracy for character:  u 0.9007633587786259
accuracy for character:  v 0.9161290322580645
accuracy for character:  w 0.664
accuracy for character:  y 0.9124087591240876
accuracy for character:  z 0.9649122807017544
"""

"""
#rnn_{}_10_noise_out00001_2c.pth, iterations_ic=0

overall loss:  tensor(0.1318, grad_fn=<DivBackward0>)
loss for character:  a tensor(0.0447, grad_fn=<DivBackward0>)
loss for character:  b tensor(0.0575, grad_fn=<DivBackward0>)
loss for character:  c tensor(0.0557, grad_fn=<DivBackward0>)
loss for character:  d tensor(0.1202, grad_fn=<DivBackward0>)
loss for character:  e tensor(0.0990, grad_fn=<DivBackward0>)
loss for character:  g tensor(0.1457, grad_fn=<DivBackward0>)
loss for character:  h tensor(0.1380, grad_fn=<DivBackward0>)
loss for character:  l tensor(0.1300, grad_fn=<DivBackward0>)
loss for character:  m tensor(0.2037, grad_fn=<DivBackward0>)
loss for character:  n tensor(0.1828, grad_fn=<DivBackward0>)
loss for character:  o tensor(0.1929, grad_fn=<DivBackward0>)
loss for character:  p tensor(0.0616, grad_fn=<DivBackward0>)
loss for character:  q tensor(0.1650, grad_fn=<DivBackward0>)
loss for character:  r tensor(0.0747, grad_fn=<DivBackward0>)
loss for character:  s tensor(0.0523, grad_fn=<DivBackward0>)
loss for character:  u tensor(0.0790, grad_fn=<DivBackward0>)
loss for character:  v tensor(0.1414, grad_fn=<DivBackward0>)
loss for character:  w tensor(0.1963, grad_fn=<DivBackward0>)
loss for character:  y tensor(0.2801, grad_fn=<DivBackward0>)
loss for character:  z tensor(0.2311, grad_fn=<DivBackward0>)
overall accuracy:  0.8305915295764789
accuracy for character:  a 0.9883040935672515
accuracy for character:  b 0.9929078014184397
accuracy for character:  c 0.8802816901408451
accuracy for character:  d 0.9554140127388535
accuracy for character:  e 0.967741935483871
accuracy for character:  g 0.8115942028985508
accuracy for character:  h 0.8188976377952756
accuracy for character:  l 0.6954022988505747
accuracy for character:  m 0.632
accuracy for character:  n 0.676923076923077
accuracy for character:  o 0.45390070921985815
accuracy for character:  p 0.9618320610687023
accuracy for character:  q 0.9516129032258065
accuracy for character:  r 0.8389830508474576
accuracy for character:  s 0.9774436090225563
accuracy for character:  u 0.8473282442748091
accuracy for character:  v 0.6258064516129033
accuracy for character:  w 0.584
accuracy for character:  y 0.927007299270073
accuracy for character:  z 0.935672514619883
"""