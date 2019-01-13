import numpy as np
np.random.seed(10)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data.getdata import getXY, getMonthX
from helpers import mae


def LSTMmodel(xtrain, ytrain):
    model = Sequential()
    model.add(LSTM(10, input_shape=(xtrain.shape[1], 1), return_sequences=False))
    model.add(Dense(10))
    model.add(Dense(4))
    #output layer
    model.add(Dense(1))
    
    #callbacks
    checkpoint = ModelCheckpoint('./models/LSTMmodel.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10)
    board = TensorBoard(log_dir='./logs')
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(xtrain, ytrain, epochs=300, batch_size=256, verbose=2,
             validation_split=0.2, callbacks=[checkpoint, earlystop])
    
    return model




x, y = getXY('./data/raw.csv')
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

hist = LSTMmodel(xtrain, ytrain) # start training
m = load_model('./models/LSTMmodel.hdf5') # load best model
ypred = m.predict(xtest)

print('mae: ', mae(ytest, ypred))










