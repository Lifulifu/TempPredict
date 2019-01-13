import numpy as np
from sklearn.model_selection import train_test_split
from data.getdata import getXY
from helpers import mae

class TrivialModel():
    def __init__(self):
        pass
    
    def predict(self, testx):
        return testx[:, -1, 0] #output last temperature of x


x, y = getXY('./data/raw.csv')
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)


m = TrivialModel()
ypred = m.predict(xtest)

print('mae: ', mae(ytest, ypred))
