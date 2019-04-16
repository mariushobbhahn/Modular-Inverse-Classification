import os.path as osp
import sys
import torch
import torch.utils.data as data
import numpy as np

class ComparisonData(data.Dataset):

    def __init__(self, classes, sequences):
        self.classes = classes
        self.sequences = sequences
        self.chars = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']


    def __len__(self):
        return (len(self.classes))

    def __getitem__(self, index):

        class_ = self.classes[index]
        sequence_ = torch.Tensor(self.sequences[index])
        onehot = torch.zeros((1, len(self.chars)))
        onehot[0][self.chars.index(class_)] = 1

        return(sequence_, onehot)






