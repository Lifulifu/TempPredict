import requests
from bs4 import BeautifulSoup
import csv
import datetime

def writeCSV(url, path, date):
    html = requests.get(url)
    root = BeautifulSoup(html.content, features="html.parser")

    table = root.select_one("#MyTable tbody")

    with open(path, 'a') as f:
        for row in table.select("tr"):
            if row.select("th") != []: #it's a header row, skip
                continue
            #write date on start of row
            f.write(date)
            f.write(",")
            #write data
            for td in row.select("td"):
                f.write(td.string.replace(u'\xa0', ''))
                f.write(',')
            f.write('\n')


def crawl(startdate, enddate):
    currdate = datetime.date(startdate[0], startdate[1], startdate[2])
    oneday = datetime.timedelta(days=1)
    count = 0

    print("start date: ", currdate.isoformat())
    while currdate <= datetime.date(enddate[0], enddate[1], enddate[2]):
        #url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467410&stname=%25E8%2587%25BA%25E5%258D%2597&datepicker={}".format(currdate.isoformat())
        #url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=C0X100&stname=%25E8%2587%25BA%25E5%258D%2597%25E5%25B8%2582%25E5%258C%2597%25E5%258D%2580&datepicker={}".format(currdate.isoformat())
        #url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=C0C480&stname=%25E6%25A1%2583%25E5%259C%2592&datepicker={}".format(currdate.isoformat())
        #url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467571&stname=%25E6%2596%25B0%25E7%25AB%25B9&datepicker={}".format(currdate.isoformat())
        url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467990&stname=%25E9%25A6%25AC%25E7%25A5%2596&datepicker={}".format(currdate.isoformat())
        writeCSV(url, "Matsu/2018_1-1_to_2018_10-31.csv", currdate.isoformat())
        print(currdate.isoformat(), " done")
        currdate = currdate + oneday
        count = count+1
    print("last date: ", (currdate-oneday).isoformat())
    print("total days: ", count)


crawl((2018, 1, 1), (2018, 10, 31))





