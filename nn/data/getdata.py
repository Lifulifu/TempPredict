import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import datetime


# helper function for write csv
def getDF(url, path, date):
    html = requests.get(url)
    root = BeautifulSoup(html.content, features="html.parser")
    table = root.select_one("#MyTable")

    df = pd.read_html(str(table))[0]
    header = df.iloc[2]
    df = df.drop(df.index[[0,1,2]])
    df.columns = header
    
    return df.reset_index(drop=True)


# crawl weather data and save as csv file
def crawlCSV(startdate, enddate, path):
    currdate = datetime.date(startdate[0], startdate[1], startdate[2])
    oneday = datetime.timedelta(days=1)
    df_list = []

    # loop every day
    while currdate <= datetime.date(enddate[0], enddate[1], enddate[2]):
        url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467410&stname=%25E8%2587%25BA%25E5%258D%2597&datepicker={}".format(currdate.isoformat())
        df_list.append(getDF(url, path, currdate.isoformat()))
        print(currdate.isoformat(), " done")
        currdate = currdate + oneday
    
    all_df = pd.concat(df_list).to_csv(path, index=False)
    print("all done.")
    return


# get weather data np arrays from csv file
def getXY(path, inputHrs=72, hrsAfter=24, features=['Temperature']):
    all_df = pd.read_csv(path)
    
    for feature in features:
        all_df[feature] = pd.to_numeric(all_df[feature], errors='coerce').interpolate() # turn 'X' into NaN
    
    x, y = [], []
    for startHr in range(all_df.shape[0]-inputHrs-hrsAfter-1):
        x.append(all_df.loc[startHr:startHr+inputHrs-1, features].values)
        y.append(all_df.loc[startHr+inputHrs+hrsAfter, 'Temperature'])
        
    x, y = np.stack(x), np.stack(y)
    print('x: ', x.shape, 'y: ', y.shape)
    return x, y


# helper function for getMonthX
def toCircular(month):
    month = month-1 # start from 0
    return [np.cos(month * 2*np.pi/12), np.sin(month * 2*np.pi/12)]

    
# get circular-encoded month data
def getMonthX(startDate, hours):
    date = datetime.datetime(year=startDate[0], month=startDate[1], day=startDate[2])
    x = []
    for i in range(hours): 
        x.append(toCircular(date.month))
        date += datetime.timedelta(hours=1)
    
    return np.array(x)
        
        

#crawlCSV((2017,1,1), (2017,12,31), 'raw.csv')

















