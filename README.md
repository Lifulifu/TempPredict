# Temperature Prediction Using NN and Xgb

## 1. Objective
We wish to predict future temperature (every hour) using past weather data.

## 2. Getting Data
We crawled our training data from [CWB Observation Data Inquire System](https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp?fbclid=IwAR03ffdzMn6oSFDsNSeT34qiOHi5ut4rmW3rIriom7PJGXeFaSqE5I9MyZg). Crawler is at `nn/data/crawler.py`, just call `crawl()` to get csv file of hour-wise weather data between given date interval. For instance:
```python
crawl((1010, 1, 1), (2018, 12, 31))
```

## 3. Processing Data
Data preprocessing helper function are in `nn/data/dataGen.py`. For instance:
```python
x, y = genXY(path, feature, inputDim, nDaysAfter)
```
reads the csv file you crawed and returns numpy arrays that can be directly used for model input and output, where `path` is the path of the csv, `inputDim` specifies how many hours of temperature you want to input, and `nDaysAfter` specifies that y should be the temperature n days after the latest input hour.

Other functions are used for different input format, depending on which model you want to use. For instance `genXYwithMon()` gives you not only temperature input, but also encoded month feature.
