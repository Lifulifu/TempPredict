import numpy as np
np.random.seed(10)
import math
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data.dataGen import genXY3, mae, mse



def genModel(name, xtrain, ytrain, kernelSize, convNeurons, denseNeurons, act, lossFunc):
    N = xtrain.shape[0]
    inputDim = xtrain.shape[1]

    model = Sequential()

    #Conv layers
    model.add(Conv1D(input_shape=(inputDim, 1), filters=convNeurons[0], kernel_size=kernelSize[0], strides=1, activation=act))
    model.add(Conv1D(filters=convNeurons[1], kernel_size=kernelSize[1], strides=1, activation=act))
    model.add(Conv1D(filters=convNeurons[2], kernel_size=kernelSize[2], strides=1, activation=act))
    model.add(Flatten())

    #Dense layers
    model.add(Dense(denseNeurons[0], activation=act))
    model.add(Dense(denseNeurons[1], activation=act))
    model.add(Dense(denseNeurons[2], activation=act))

    #output
    model.add(Dense(1))

    #set callbacks
    checkpoint = ModelCheckpoint('./World/' + name + '.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=30)
    
    #fit model
    model.compile(loss='mse', optimizer=Adam())
    print(model.summary())
    history = model.fit(x=xtrain, y=ytrain, validation_split=0.2,
                        epochs=500, batch_size=300, verbose=2, callbacks=[checkpoint, earlystop])
    return history

