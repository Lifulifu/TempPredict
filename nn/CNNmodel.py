import numpy as np
np.random.seed(10)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data.getdata import getXY
from helpers import mae


def CNNmodel(xtrain, ytrain):
    N = xtrain.shape[0]
    inputDim = xtrain.shape[1]

    model = Sequential()

    #Conv layers
    model.add(Conv1D(input_shape=(inputDim, 1), filters=4, kernel_size=1, strides=1, activation='relu'))
    model.add(Conv1D(filters=8, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv1D(filters=12, kernel_size=3, strides=1, activation='relu'))
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
    earlystop = EarlyStopping(monitor='val_loss', patience=20)
    
    #fit model
    model.compile(loss='mse', optimizer=Adam())
    print(model.summary())
    history = model.fit(x=xtrain, y=ytrain, validation_split=0.2, shuffle=True, 
                        epochs=300, batch_size=200, verbose=2, callbacks=[board, checkpoint, earlystop])

    return history


x, y = getXY('./data/raw.csv')
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

hist = CNNmodel(xtrain, ytrain) # start training
m = load_model('./models/CNNmodel.hdf5') # load best model
ypred = m.predict(xtest)

print('mae: ', mae(ytest, ypred))



















