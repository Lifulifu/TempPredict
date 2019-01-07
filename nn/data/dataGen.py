import numpy as np
import datetime
import pandas as pd

DATE = 0
PRES = 2
TEMP = 4
HUMD = 6

def readFromCSV(path):
    return np.genfromtxt(path, delimiter=',')
    

def genXYwithMon(path, inputDim, nDaysAfter):
    data = readFromCSV(path) #shape: 70152*19
    N = data.shape[0] - nDaysAfter - inputDim #data amount
    
    # patch nan
    count = 0
    for i in range(len(data)):
        if np.isnan(data[i, TEMP]):
            count = count+1
            j = i
            while np.isnan(data[j, TEMP]):
                j = j+1
            data[i, TEMP] = data[j, TEMP]
    print('nan count:', count)
    
    x, xaux, y = [], [], []
    print('date', data[2, DATE])
    for i in range(N):
        date = datetime.datetime.strptime(data[i, DATE], '%Y-%m-%d')
        cirMonth = toCircular(date.month)
        
        x.append(data[i:i+inputDim, TEMP])
        xaux.append(cirMonth)
        y.append(data[i+inputDim+nDaysAfter, TEMP])
        
        if np.isnan(data[i+inputDim+nDaysAfter, TEMP]):
            print('nana')
    
    x, y = np.array(x), np.array(y)
    x = x[:,:,np.newaxis]

    return x, xaux, y


def genXY(path, feature, inputDim, nDaysAfter):
    data = pd.read_csv(path, delimiter=',')
    col = data[feature].interpolate()
    N = data.shape[0] - nDaysAfter - inputDim #x data amount
    
    x, y = [], []
    for i in range(N):
        x.append(col[i:i+inputDim])
        y.append(col[i+inputDim+nDaysAfter])
    
    x, y = np.array(x), np.array(y)
    
    return x, y


def toCircular(month):
    month = month-1 # start from 0
    return (np.cos(month * 2*np.pi/12), np.sin(month * 2*np.pi/12))


def genXY1Year(path, inputDim, nDaysAfter):
    data = readFromCSV(path) #shape: 70152*19
    N = 24*365
    
    x, y = [], []
    for i in range(N):
        yindex = data.shape[0]-i-1
        x.append(data[yindex-nDaysAfter-inputDim : yindex-nDaysAfter, TEMP])
        y.append(data[yindex, TEMP])
    
    x, y = np.array(x), np.array(y)
    x = x[:,:,np.newaxis]

    return x, y


def genXY3(path, inputDim, nDaysAfter): #gen 3-channel input
    data = readFromCSV(path) #shape: 70152*19
    N = data.shape[0] - nDaysAfter - inputDim #data amount
    print('original data: ', N)
    
    x, y = [], []
    for i in range(N):
        baddata = False
        xrow = []
        for j in range(inputDim):
            hourdata = [data[i+j, TEMP], data[i+j, PRES], data[i+j, HUMD]]
            xrow.append(hourdata)
            for xd in hourdata:
                if np.isnan(xd):
                    baddata = True
            if baddata:
                break
                
        if not baddata:
            x.append(xrow)
            y.append(data[i+inputDim+nDaysAfter, TEMP])
    
    x, y = np.array(x), np.array(y)
    N = x.shape[0]
    print('useable data: ', N)

    #trx, try, tex, tey
    return x[:N-8760, :, :], y[:N-8760], x[N-8760:, :, :], y[N-8760:]


def mae(a, b):
    if a.shape[0] != b.shape[0]:
        return -1
    N = a.shape[0]
    absSum = 0
    for n in range(N):
        absSum = absSum + abs(a[n] - b[n])
    return absSum / N


def mse(a, b):
    N = a.shape[0]
    sqrSum = 0
    for n in range(N):
        sqrSum = sqrSum + (a[n] - b[n])**2
    return sqrSum / N

    
    