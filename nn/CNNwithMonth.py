import numpy as np
np.random.seed(10)
from keras.models import Model, load_model
from keras.layers import Dense, Input, concatenate, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data.getdata import getXY, getMonthX
from helpers import mae


def CNNwithMonth(xtrain, xaux, ytrain):

    tempInput = Input(shape=(xtrain.shape[1], 1), name='tempInput')
    monthInput = Input(shape=(2,), name='monthInput')
    
    #conv layers
    conv1out = Conv1D(input_shape=(xtrain.shape[1], 1), filters=19, kernel_size=7, strides=1, activation='relu')(tempInput)
    conv2out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv1out)
    conv3ut = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv2out)
    flatOut = Flatten()(conv3ut)
    
    #concat
    dense1in = concatenate([flatOut, monthInput])
    
    #dense
    dense1out = Dense(30)(dense1in)
    dense2out = Dense(26)(dense1out)
    dense3out = Dense(22)(dense2out)
    
    #output
    out = Dense(1, name='out')(dense3out)
    
    #define model i/o
    model = Model(inputs=[tempInput, monthInput], outputs=[out])
    print(model.summary())
    
    #set callbacks
    board = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint('./models/CNN_mon.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10)
    
    #fit model
    model.compile(loss='mse', optimizer=Adam())
    #plot_model(model, to_file='plot/CNN_func.png')
    
    history = model.fit(
        {'tempInput': xtrain, 'monthInput': xaux}, 
        {'out': ytrain},
        validation_split=0.2,epochs=500, batch_size=400, verbose=2, 
        callbacks=[checkpoint, earlystop]
    )
    
    return model


x, y = getXY('./data/raw.csv')
x2 = getMonthX((2010,1,1), x.shape[0])

# fix random_state so that index of x will correspond to x2
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=87) 
x2train, x2test, _, _ = train_test_split(x2, y, test_size=0.25, random_state=87)

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
print(x2train.shape, x2test.shape)


#hist = CNNwithMonth(xtrain, x2train, ytrain) # start training
m = load_model('./models/CNN_mon.hdf5') # load best model
ypred = m.predict([xtest, x2test])

print('mae: ', mae(ytest, ypred))












