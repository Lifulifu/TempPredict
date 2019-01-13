import numpy as np
np.random.seed(10)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, concatenate, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data.getdata import getXY
from helpers import mae



def CNNsepFeature(xtrain, ytrain):
    
    #inputs
    tempIn = Input(shape=(72, 1), name='tempInput')
    humdIn = Input(shape=(72, 1), name='humdInput')
    presIn = Input(shape=(72, 1), name='presInput')

    #conv for temp
    conv1out = Conv1D(input_shape=(72, 1), filters=19, kernel_size=7, strides=1, activation='relu')(tempIn)
    conv2out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv1out)
    conv3out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv2out)
    flatOut1 = Flatten()(conv3out)
    
    #conv for humd
    conv1out = Conv1D(input_shape=(72, 1), filters=19, kernel_size=7, strides=1, activation='relu')(humdIn)
    conv2out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv1out)
    conv3out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv2out)
    flatOut2 = Flatten()(conv3out)
    
    #conv for pres
    conv1out = Conv1D(input_shape=(72, 1), filters=19, kernel_size=7, strides=1, activation='relu')(presIn)
    conv2out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv1out)
    conv3out = Conv1D(filters=38, kernel_size=7, strides=1, activation='relu')(conv2out)
    flatOut3 = Flatten()(conv3out)
    
    #concat
    dense1in = concatenate([flatOut1, flatOut2, flatOut3])
    
    #dense
    dense1out = Dense(30)(dense1in)
    dense2out = Dense(26)(dense1out)
    dense3out = Dense(22)(dense2out)
    
    #output
    out = Dense(1, name='out')(dense3out)
    
    #define model i/o
    model = Model(inputs=[tempIn, humdIn, presIn], 
                  outputs=[out])
    print(model.summary())
    
    #set callbacks
    board = TensorBoard(log_dir='./logs/sepCNN')
    checkpoint = ModelCheckpoint('./models/CNNsepFeature.hdf5', monitor='val_loss', 
                                 save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10)
    
    #fit model
    model.compile(loss='mse', optimizer=Adam())
    
    history = model.fit(
        {'tempInput': xtrain[0], 'humdInput': xtrain[1], 'presInput': xtrain[2]}, 
        {'out': ytrain},
        validation_split=0.2,epochs=300, batch_size=400, verbose=2, 
        callbacks=[checkpoint, earlystop, board]
    )
    
    return history



x, y = getXY('./data/raw.csv', features=['Temperature', 'Td dew point', 'StnPres'])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

xtrainin = [xtrain[:,:,0][:,:,np.newaxis],xtrain[:,:,1][:,:,np.newaxis],xtrain[:,:,2][:,:,np.newaxis]]
xtestin = [xtest[:,:,0][:,:,np.newaxis],xtest[:,:,1][:,:,np.newaxis],xtest[:,:,2][:,:,np.newaxis]]
print('xtrain after sep feature: ', xtrainin[0].shape, xtrainin[1].shape, xtrainin[2].shape)


hist = CNNsepFeature(xtrainin, ytrain) # start training
m = load_model('./models/CNNsepFeature.hdf5') # load best model
ypred = m.predict(xtestin)

print('mae: ', mae(ytest, ypred))












