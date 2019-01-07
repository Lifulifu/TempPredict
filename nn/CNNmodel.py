import numpy as np
np.random.seed(10)
import math
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from data.dataGen import genXY3

def Model(xtrain, ytrain):
    N = xtrain.shape[0]
    inputDim = xtrain.shape[1]

    model = Sequential()

    #Conv layers
    model.add(Conv1D(input_shape=(inputDim, 3), filters=4, kernel_size=1, strides=1, activation='relu'))
    model.add(Conv1D(filters=8, kernel_size=5, strides=1, activation='relu'))
    model.add(Conv1D(filters=12, kernel_size=5, strides=1, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())

    #Dense layers
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(4, activation='relu'))
    #model.add(Dropout(0.1))

    #output
    model.add(Dense(1))

    #set callbacks
    board = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint('./models/CNNmodel.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    
    #fit model
    print(model.summary())
    model.compile(loss='mse', optimizer=Adam())
    history = model.fit(x=xtrain, y=ytrain, validation_split=0.2, shuffle=True, 
                        epochs=300, batch_size=200, verbose=2, callbacks=[board, checkpoint])

    return model

