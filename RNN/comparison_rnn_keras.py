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
(x_train, y_train), (x_test, y_test) = character_trajectories.load_data("../RNN/data/char_trajectories_{}x1_types_centers_only.pkl".format(dim))

"""remove the padding of the x_data"""
x_train = x_train[:,0,:]
x_test = x_test[:,0,:]

if types or sparse:
    """reduce the types to characters by taking the max along the axis"""
    print("test sample before the type reduction: ", x_train[0])
    x_train = np.amax(x_train, axis=2)
    x_test = np.amax(x_test, axis=2)
    print("test sample after type reduction: ", print(x_train[0]))


print("x_train shape: " ,x_train.shape)
print("y_train shape: " ,y_train.shape)
print("x_test shape: " ,x_test.shape)
print("y_test shape: " ,y_test.shape)


# Get your input dimensions

INPUT_SHAPE=(205, 2)

# Output dimensions is the shape of a single output vector
output_dim = len(x_train[0])
print("output_dim: ", output_dim)



RNN_FILE_MODEL = os.path.join(DIR_MODEL, "comparison_model_{}_sparse1_v{}.hdf5".format(dim,i))
hidden_size = 205
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
        validation_split=0.2,
        callbacks=[checkpointer],
        verbose=0
        )