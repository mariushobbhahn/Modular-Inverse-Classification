import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from process_data import pad_sequence, cumulative_sum_seq, plot_sequence, standardize_data, pad_class, char_to_one_hot, cluster_chars, cluster_to_one_hot, pad_clustered_char, pad_entire_sequence, save_obj, load_obj

#main function:
def load_character_trajectories_handpicked(pad_sequences=True,
                                        cumulative_sum=True,
                                        gen_sequences=True,
                                        standardize=True,
                                        filename='',
                                        ):

    #load data
    mat_contents = sio.loadmat('data/mixoutALL_shifted.mat')
    consts = mat_contents['consts']
    sequences = mat_contents['mixout'][0]


    #we want all the sequences to have the same length,
    #so we add zeros for padding in the end.
    MAX_LEN = max([len(seq[0]) for seq in sequences])

    if pad_sequences:
        sequences_final = np.zeros((2858, 2, MAX_LEN))

        #pad dimension x and y
        for i in range(len(sequences)):
            sequences_final[i][0] = pad_sequence(sequences[i][0], MAX_LEN)
            sequences_final[i][1] = pad_sequence(sequences[i][1], MAX_LEN)

        #print result to show it has been padded
        #print("shape of sequences final", np.shape(sequences_final))


    if cumulative_sum:
        sequences_final = [cumulative_sum_seq(seq) for seq in sequences_final]

    if gen_sequences:
    #swap axis such that shape is (205,2) not (2,205)
        #print("shape before swap: ",np.shape(sequences_final))
        sequences_final = np.swapaxes(sequences_final, axis1=1, axis2=2)
        #print("shape after swap: ",np.shape(sequences_final))

    #plot random letter before standardizing
    #plot_sequence(300, sequences_final, swapaxis=True)
    if standardize:
        sequences_final = standardize_data(sequences_final, separate=True)
        #plot same letter after standardizing
        #plot_sequence(300, sequences_final, swapaxis=True)

    #now we prepare classes:
    classes = []
    #index of last letter of kind
    classes.extend(['a'] * (97 - 0))#a = 96
    classes.extend(['b'] * (170 - 97))#b = 169
    classes.extend(['c'] * (225 - 170))#c = 224
    classes.extend(['d'] * (307 - 225))#d = 306
    classes.extend(['e'] * (420 - 307))#e = 419
    classes.extend(['g'] * (486 - 420))#g = 485
    classes.extend(['h'] * (543 - 486))#h = 542
    classes.extend(['l'] * (623 - 543))#l = 622
    classes.extend(['m'] * (692 - 623))#m = 691
    classes.extend(['n'] * (748 - 692))#n = 747
    classes.extend(['o'] * (816 - 748))#o = 815
    classes.extend(['p'] * (886 - 816))#p = 885
    classes.extend(['q'] * (956 - 886))#q = 955
    classes.extend(['r'] * (1013 - 956))#r = 1012
    classes.extend(['s'] * (1077 - 1013))#s = 1076
    classes.extend(['u'] * (1144 - 1077))#u = 1143
    classes.extend(['v'] * (1218 - 1144))#v = 1217
    classes.extend(['w'] * (1278 - 1218))#w = 1277
    classes.extend(['y'] * (1345 - 1278))#y = 1344
    classes.extend(['z'] * (1433 - 1345))#z = 1432
    classes.extend(['a'] * (1507 - 1433))#a = 1506
    classes.extend(['b'] * (1575 - 1507))#b = 1574
    classes.extend(['c'] * (1662 - 1575))#c = 1661
    classes.extend(['d'] * (1737 - 1662))#d = 1736
    classes.extend(['e'] * (1810 - 1737))#e = 1809
    classes.extend(['g'] * (1882 - 1810))#g = 1881
    classes.extend(['h'] * (1952 - 1882))#h = 1951
    classes.extend(['l'] * (2046 - 1952))#l = 2045
    classes.extend(['m'] * (2102 - 2046))#m = 2101
    classes.extend(['n'] * (2176 - 2102))#n = 2175
    classes.extend(['o'] * (2249 - 2176))#o = 2248
    classes.extend(['p'] * (2310 - 2249))#p = 2309
    classes.extend(['q'] * (2364 - 2310))#q = 2363
    classes.extend(['r'] * (2426 - 2364))#r = 2425
    classes.extend(['s'] * (2495 - 2426))#s = 2494
    classes.extend(['u'] * (2558 - 2495))#u = 2558
    classes.extend(['v'] * (2639 - 2558))#v = 2639
    classes.extend(['w'] * (2704 - 2639))#w = 2704
    classes.extend(['y'] * (2774 - 2704))#y = 2774
    classes.extend(['z'] * (2857 - 2774))#z = 2857


    #these are all possible letters:
    list_of_chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

    """
    for i in range(len(classes)):
        plot_sequence(index=i, title='index_{} '.format(i), filename='char_{}_index_{}'.format(classes[i], i), save=True, plot=False, sequences=sequences_final, swapaxis=True)
        print("class_{}: ".format(i), classes[i])
    """

    #for i in range(1425, 1510):
    #    plot_sequence(index=i, title='index_{} '.format(i), sequences=sequences_final, swapaxis=True)
    handpicked_chars = [
        [68, 1462, 89, 1492], #a
        [99, 109, 1519, 1572], #b
        [175,182, 217, 1623], #c
        [226, 232, 258, 1692], #d
        [312, 319, 352, 1791], #e
        [421, 444, 485, 1869], #g
        [500, 530, 1933, 1949], #h
        [547, 574, 567, 1996], #l
        [628, 638, 2062, 2082], #m
        [700, 715, 742, 2131], #n
        [751, 779, 810, 2207], #o
        [825, 834, 2261, 2301], #p
        [889, 902, 2314, 2342], #q
        [959, 986, 2371, 2385], #r
        [1016, 1070, 2493, 2494], #s
        [1084, 1104, 2507, 2546], #u
        [1144, 1180, 2595, 2629], #v
        [1221, 1264, 2644, 2677], #w
        [1284, 1322, 2707, 2740], #y
        [1345, 1364, 2786, 2855] #z
    ]


    #this must have 20 items of 4 pairings
    handpicked_class_seq_mappings = []
    for i in range(20):
        list_of_tuples = []
        for j in range(4):
            type = np.zeros(4)
            type[j] = 1
            sequence = sequences_final[handpicked_chars[i][j]]
            list_of_tuples.append((type, sequence))

        handpicked_class_seq_mappings.append(np.array(list_of_tuples))

    #print(handpicked_class_seq_mappings)

    # create a dictionary with letter to sequence mapping
    dictionary = dict(zip(list_of_chars, handpicked_class_seq_mappings))

    #plot all sequences and save them in data/types_images
    for i in range(20):
        letter = list_of_chars[i]
        for j in range(4):
            test_seq = dictionary[letter][j][1]
            test_seq = np.reshape(test_seq, (1, 205, 2))
            filename_image = 'data/types_images/' + 'types_' + letter + '_' + str(j) + '.png'
            print(filename_image)
            plot_sequence(index=0, sequences=test_seq, swapaxis=True, filename=filename_image, plot=False, save=True)

    #plot_sequence(index=0, sequences=test_seq, swapaxis=True)
    all_data = dictionary


    #save dict as numpy file
    np.save(filename, all_data)
    print("saved at: ", filename)


if __name__ == '__main__':

    load_character_trajectories_handpicked(filename='data/sequences_4_handpicked.npy')