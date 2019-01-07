import numpy as np
np.random.seed(10)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data.dataGen import genXY, mae, mse


def LSTMmodel(xtrain, ytrain):
    model = Sequential()
    model.add(LSTM(10, input_shape=(xtrain.shape[1], 1), return_sequences=True))
    model.add(LSTM(10, return_sequences=False))
    #output layer
    model.add(Dense(1))
    
    #callbacks
    checkpoint = ModelCheckpoint('./models/LSTMmodel.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=40)
    board = TensorBoard(log_dir='./logs')
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(xtrain, ytrain, epochs=300, batch_size=128, verbose=2,
             validation_split=0.2, callbacks=[checkpoint, earlystop])
    
    return model



