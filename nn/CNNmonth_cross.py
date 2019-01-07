import numpy as np
from keras.models import load_model
from data.dataGen import pandasXY, mae
from CNNwithMonth import CNNFunc


X, Xaux, Y = pandasXY('../data/2010_to_2018.csv', 72, 24)
X = np.split(X, 8)
Xaux = np.split(Xaux, 8)
Y = np.split(Y, 8)

for testYr in range(8):
    print('Year 201'+str(testYr)+' as test.', X[testYr].shape)
    
    # 7 yr train data
    trainx = [X[yr] for yr in range(8) if yr != testYr]
    trainx = np.concatenate(tuple(trainx), axis=0)
    trainxaux = [Xaux[yr] for yr in range(8) if yr != testYr]
    trainxaux = np.concatenate(tuple(trainxaux), axis=0)
    trainy = [Y[yr] for yr in range(8) if yr != testYr]
    trainy = np.concatenate(tuple(trainy), axis=0)
    
    # 1 yr test data
    testx, testxaux, testy = X[testYr], Xaux[testYr], Y[testYr]
    
    print(trainx.shape, trainxaux.shape, trainy.shape)
    
    CNNFunc(trainx, trainxaux, trainy, testYr)

    

