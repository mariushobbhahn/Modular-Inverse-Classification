import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(sequence, title, swapaxis=False, deltas=False):
    if swapaxis:
        sequence = np.swapaxes(sequence,0 ,1)
    if deltas:
        x,y = np.cumsum(sequence[0][0]), np.cumsum(sequence[0][1])
    else:
        x,y = sequence[0], sequence[1]
    plt.title(title)
    plt.plot(x, y)
    plt.show()