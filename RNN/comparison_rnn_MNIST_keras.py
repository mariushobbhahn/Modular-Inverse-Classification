from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten
#from data import character_trajectories
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from keras.datasets import mnist

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "weights")
#RNN_FILE_MODEL_weights = os.path.join(DIR_MODEL, "weights_RNN_v1.hdf5")

#types = False
#sparse = True
dim = 10
# load training and validation data
#x_train, y_train, x_val, y_val, x_test, y_test = np.load("data/sequences_all_onehot.npy")
(x_train, y_train), (_, _) = mnist.load_data()


x_train = x_train.reshape(len(x_train), -1, 1)
y_train = y_train.reshape(len(y_train), -1)

print("x_train shape: " , np.shape(x_train))
print("y_train shape: " , np.shape(y_train))
#print("x_val shape: " , np.shape(x_val))
#print("y_val shape: " , np.shape(y_val))
#print("x_test shape: " , np.shape(x_test))
#print("y_test shape: " , np.shape(y_test))


# Get your input dimensions

INPUT_SHAPE=(np.shape(x_train[0]))
print("input shape: ", INPUT_SHAPE)

# Output dimensions is the shape of a single output vector
output_dim = len(y_train[0])
print("output_dim: ", output_dim)


i = 1

RNN_FILE_MODEL = os.path.join(DIR_MODEL, "comparison_model_{}_mnist_v{}.hdf5".format(dim,i))
hidden_size = 200
"""create model"""
rnn = Sequential()
rnn.add(LSTM(hidden_size, return_sequences=False, input_shape=INPUT_SHAPE))
rnn.add(Dense(units=output_dim, activation='softmax'))

rnn.compile(loss='mean_squared_error', optimizer='adam')
rnn.summary()

checkpointer = ModelCheckpoint(filepath=RNN_FILE_MODEL, verbose=1, save_best_only=True)
rnn.fit(x=x_train,
        y=y_train,
        epochs=10000,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        callbacks=[checkpointer],
        verbose=1
        )