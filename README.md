# Temperature Prediction Using NN and Xgb

## 1. Objective
We wish to predict future temperature (every hour) using past weather data.

## 2. Training Data
We crawled our training data from [CWB Observation Data Inquire System](https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp?fbclid=IwAR03ffdzMn6oSFDsNSeT34qiOHi5ut4rmW3rIriom7PJGXeFaSqE5I9MyZg). Crawler is at `nn/data/crawler.py`, just call `crawl()` to get csv file of hour-wise weather data between given date interval. For instance:
```python
crawl((1010, 1, 1), (2018, 12, 31))
```


