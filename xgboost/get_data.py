import os
#from os import walk
#from os import join
import numpy as np
import pandas as pd
import time
import datetime
from datetime import timedelta
import multiprocessing as mp

def from_2010(path, tag, train_hour, max_train_hour, pred_hour):
    data = pd.read_csv(path, header = None)
    
    sets = []

    offset = max_train_hour + pred_hour
    
    label = data.iloc[offset:, [tag]]
    temp = data.iloc[:, [tag]].reset_index(drop = True).copy()
    temp.columns = [0]
    
    for i in range(train_hour-1):
        temp[i+1] = temp.iloc[i+1:, [0]].reset_index(drop = True)
    temp = temp.iloc[:-offset, :].reset_index(drop = True)

    print (temp.shape)

    sets = [temp, label]
    return sets


def get_time(path, max_hour, pred_hour):
    data = pd.read_csv(path, header = None)

    offset = max_hour + pred_hour
    data[0] = pd.to_datetime(data[0], format = '%Y-%m-%d')
    time = data.iloc[:, [1]]
    date = data.iloc[:, [0]].copy()
    
    
    pool = mp.Pool()
    result = [pool.apply_async(deal_time, (date.iloc[i, 0],)) for i in range(data.shape[0])]
    #dates = [res.get() for res in result]
    #print (dates)
    a_date = pd.DataFrame([res.get() for res in result])

    """
    for i in range(data.shape[0]):
        #pool.apply_async(deal_time, args = (data, date, i))
        #deal_time(data, date, i)
        first_day = datetime.datetime(year = data.iloc[i, 0].year, month = 1, day = 1, hour = data.iloc[i, 0].hour, minute = 0, second = 0)
        #first_hour = datetime.datetime(year = data.iloc[i, 0].year, month = data.iloc[i, 0].month, day = data.iloc[i, 0].day, hour = 0, minute = 0, second = 0)
        date.iloc[i, 0] = (date.iloc[i, 0] - first_day).total_seconds() / 3600 / 24 / 366
    """
    time /= 24
    time = time.iloc[offset:, [0]].reset_index(drop = True)
    a_date = a_date.iloc[offset:, [0]].reset_index(drop = True)
    print (time.shape)
    print (a_date.shape)
    
    return [a_date, time]

def deal_time(date):
    #print (i)
    #offset = end - start
    #for i in range(start, end):
    first_day = datetime.datetime(year = date.year, month = 1, day = 1, hour = date.hour, minute = 0, second = 0)
    #first_hour = datetime.datetime(year = data.iloc[i, 0].year, month = data.iloc[i, 0].month, day = data.iloc[i, 0].day, hour = 0, minute = 0, second = 0)
    date = (date - first_day).total_seconds() /3600/24/366
    
    return date

"""
def deal_time(data, date, i, start, end):
    #print (i)
    offset = end - start
    for i in range(start, end):
        first_day = datetime.datetime(year = data.iloc[i, 0].year, month = 1, day = 1, hour = data.iloc[i, 0].hour, minute = 0, second = 0)
        first_hour = datetime.datetime(year = data.iloc[i, 0].year, month = data.iloc[i, 0].month, day = data.iloc[i, 0].day, hour = 0, minute = 0, second = 0)
        date.iloc[i, 0] = (date.iloc[i, 0] - first_day).total_seconds() / 3600 / 24 / 366
    return date
"""
if __name__ == '__main__':
    now = time.time()
    test = get_time('North/2010_1-1_to_2017_12-31.csv', 72, 24)
    #print (test[0])
    print (time.time() - now)


