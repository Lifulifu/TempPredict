import numpy as np
np.random.seed(10)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, concatenate, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data.getdata import getXY
from helpers import mae



def CNNconcatFeature(xtrain, ytrain):
    N = xtrain.shape[0]
    inputDim = 72*3

    model = Sequential()

    #Conv layers
    model.add(Conv1D(input_shape=(inputDim, 1), filters=19, kernel_size=5, strides=1, activation='relu'))
    model.add(Conv1D(filters=38, kernel_size=5, strides=1, activation='relu'))
    model.add(Conv1D(filters=38, kernel_size=5, strides=1, activation='relu'))
    model.add(Flatten())

    #Dense layers
    model.add(Dense(30, activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(22, activation='relu'))

    #output
    model.add(Dense(1))

    #set callbacks
    board = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint('./models/CNNconcatFeature.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=100)
    
    #fit model
    print(model.summary())
    model.compile(loss='mse', optimizer=Adam())
    history = model.fit(x=xtrain, y=ytrain, validation_split=0.2, shuffle=True, 
                        epochs=500, batch_size=400, verbose=1, callbacks=[board, checkpoint, earlystop])

    return history


x, y = getXY('./data/raw.csv', features=['Temperature', 'Td dew point', 'StnPres'])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

xtraincat = np.concatenate((xtrain[:,:,0],xtrain[:,:,1],xtrain[:,:,2]), axis=1)[:,:,np.newaxis]
xtestcat = np.concatenate((xtest[:,:,0],xtest[:,:,1],xtest[:,:,2]), axis=1)[:,:,np.newaxis]
print('x after concat: ', xtraincat.shape, xtestcat.shape)

hist = CNNconcatFeature(xtraincat, ytrain) # start training
m = load_model('./models/CNNconcatFeature.hdf5') # load best model
ypred = m.predict(xtestcat)

print('mae: ', mae(ytest, ypred))












