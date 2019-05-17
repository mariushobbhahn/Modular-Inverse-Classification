from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten
from data import character_trajectories
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
#RNN_FILE_MODEL_weights = os.path.join(DIR_MODEL, "weights_RNN_v1.hdf5")

types = False
sparse = True
dim = 20
# load training and validation data
x_train, y_train, x_val, y_val, x_test, y_test = np.load("data/sequences_all_onehot.npy")

print("x_train shape: " , np.shape(x_train))
print("y_train shape: " , np.shape(y_train))
print("x_val shape: " , np.shape(x_val))
print("y_val shape: " , np.shape(y_val))
print("x_test shape: " , np.shape(x_test))
print("y_test shape: " , np.shape(y_test))


# Get your input dimensions

INPUT_SHAPE=(205, 2)

# Output dimensions is the shape of a single output vector
output_dim = len(x_train[0])
print("output_dim: ", output_dim)



RNN_FILE_MODEL = os.path.join(DIR_MODEL, "comparison_model_{}_sparse1_v{}.hdf5".format(dim,i))
hidden_size = 200
"""create model"""
rnn = Sequential()
rnn.add(LSTM(hidden_size, return_sequences=False, input_shape=INPUT_SHAPE))
rnn.add(Dense(units=output_dim, activation='softmax'))

rnn.compile(loss='mean_squared_error', optimizer='adam')
rnn.summary()

checkpointer = ModelCheckpoint(filepath=RNN_FILE_MODEL, verbose=1, save_best_only=False)
rnn.fit(x=y_train,
        y=x_train,
        epochs=10000,
        batch_size=10,
        shuffle=True,
        validation_data=(y_val, x_val),
        callbacks=[checkpointer],
        verbose=0
        )