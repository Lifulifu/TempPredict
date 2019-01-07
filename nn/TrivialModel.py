import numpy as np
from data.dataGen import genXY, mae

class TrivialModel():
    def __init__(self):
        pass
    
    def predict(self, testx):
        return testx[:, -1, 0] #output last temperature of x


# mae: 1.92