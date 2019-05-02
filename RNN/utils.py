import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(sequence, title, show=True, save=False, filename='', swapaxis=False, deltas=False):
    if swapaxis:
        sequence = np.swapaxes(sequence,0 ,1)
    if deltas:
        x,y = np.cumsum(sequence[0][0]), np.cumsum(sequence[0][1])
    else:
        x,y = sequence[0], sequence[1]
    plt.title(title)
    plt.plot(x, y)
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
    plt.clf()

def plot_pred_target(pred, target, show=True, save=True, filename='test.png'):
    pred = np.swapaxes(pred, 0,1)
    target = np.swapaxes(target, 0, 1)
    x_pred, y_pred = pred[0], pred[1]
    x_target, y_target = target[0], target[1]
    fig, ax = plt.subplots(2)
    #ax[0] = plt.subplot(211)
    ax[0].plot(x_pred, y_pred)
    ax[0].set_title('prediction')
    ar_pred = (np.max(x_pred) - np.min(x_pred))/(np.max(y_pred) - np.min(y_pred))
    ax[0].set_aspect(ar_pred)
    #ax[1] = plt.subplot(212)
    ax[1].plot(x_target, y_target)
    ax[1].set_title('target')
    ar_target = (np.max(x_target) - np.min(x_target)) / (np.max(y_target) - np.min(y_target))
    ax[1].set_aspect(ar_target)
    fig.suptitle(filename)
    fig.tight_layout()
    if save:
        plt.savefig(filename) #, bbox_inches="tight")
        fig.savefig(filename) #, bbox_inches="tight")
    if show:
        plt.show()
    fig.clf()