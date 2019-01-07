import pandas as pd
import numpy as np
from sklearn import preprocessing as pre


def inserting(data):
    for i in range(data.shape[0]):
        if data.iloc[i, 0] < -70:
            j = 0
            #try:
            while data.iloc[i+j, 0] < -70:
                j += 1
            #except:
            #    break
            
            if j < 10 :
                step = (data.iloc[i+j, 0] - data.iloc[i-1, 0]) / (j+1)

                for k in range(j):
                    data.iloc[i+k, 0] = data.iloc[i+k-1, 0] + step
                i = i +j - 1

            else:
                while data.iloc[i, 0] < -70:
                    end = 0
                    offset_start = 1
                    #try:
                    while data.iloc[i-24*offset_start, 0] < -70:
                        offset_start += 1
                    #except:
                    #    i += 1
                    #    continue
                    
                    start = i - 24 * offset_start
                    offset_end = 1
                    #try:
                    while data.iloc[i+24*offset_end, 0] <= -70:
                        offset_end += 1
                    #except:
                        #print (str)
                    #    i += 1
                    #    continue

                    end = i + 24*offset_end
                    step = (data.iloc[end, 0] - data.iloc[start, 0]) / (offset_start + offset_end)
                    data.iloc[i, 0] = data.iloc[start, 0] + step
                    i += 1

    return data

path = 'Tamsui/2018_1-1_to_2018_10-31.csv'

data = pd.read_csv(path, header=None)

for i in range(data.shape[0]):
#for i in range(100):
    #print (i)
    if data[2][i] == '/' or data[2][i] == 'X':
        data[2][i] = None
    if data[4][i] == '/' or data[4][i] == 'X':
        data[4][i] = None
    if data[6][i] == '/' or data[6][i] == 'X':
        data[6][i] = None
    

data.fillna((-99), inplace = True)
#print (data)
data.iloc[:, [2]] = inserting(data.iloc[:, [2]].astype(float))
data.iloc[:, [4]] = inserting(data.iloc[:, [4]].astype(float))
data.iloc[:, [6]] = inserting(data.iloc[:, [6]].astype(float))

data.to_csv(path, header = None, index = None)

