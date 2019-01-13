# Temperature Prediction Using NN and Xgb

## 1. Objective
We wish to predict future temperature (every hour) using past weather data.

## 2. Crawl Data
We crawled our training data from [CWB Observation Data Inquire System](https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp?fbclid=IwAR03ffdzMn6oSFDsNSeT34qiOHi5ut4rmW3rIriom7PJGXeFaSqE5I9MyZg). Crawler is at `nn/data/getdata.py`, just call `crawl(startDate, endDate, path)` to get csv file of hour-wise weather data given date interval. For instance:
```python
crawlCSV((2010, 1, 1), (2018, 12, 31), 'raw.csv')
```

## 3. Process Data
Now that we have our raw csv data, we can call `getXY(path, inputDays, hrsAfter, features)` in `nn/data/getdata.py` to get numpy arrays that can be directly used for model input and output, where `path` is the csv file you wish to read in, `inputDays` specifies how many hours you want to input, `hrsAfter` specifies how many hours later you wish to predict after the last input hour, and `features` is an array that tells the features to input in one hour. For instance:
```python
x, y = getXY('data/raw.csv', inputHrs=72, hrsAfter=24, features=['Temperature','StnPres'])
```
returns numpy arrays `x, y` with shape `(N, 72, 2), (N, )`, where `N` is the amount of your data.

For Xgb, data preprocessing is in `xgboost/get_data.py`. Use
```python
from_2010(path, tag, train_hour, max_train_hour, pred_hour)
```
to get raw feature e.g. temperature. Tag is the column in number.
```python
get_time(path, max_hour, pred_hour)
```
to return date and time been normalized by 0~1, date is divided by 366, time is devided by 24.


## 4. Models
In `nn/`, each .py file defines a model. Models are defined in functions, once the function is called, for example:
```python
CNNmodel(xtrain, ytrain)
```
Once called, the model immediately starts to fit. Each model may have different shapes of `xtrain`.

Note that, after training is done, it only returns loss history of the model. This is because we use **ModelCheckpoint** and **EarlyStopping** to save the best model to `nn/models/`, preventing from getting overfit models. Use `m = load_model(path)` to get the trained model.

* **CNNmodel:** Basic CNN with no pooling. Inputs only `Temperature` as feature

* **CNNmodel_v2:** CNN that inputs 3 features: `Temperature`, `Td dew point` and `StnPres`

* **CNNconcatFeature:** Inputs the 3 features above, but concatenated

* **CNNsepFeature:** Inputs 3 features into 3 seperated CNN, then concat them in dense layer. As shown in figure (a)

* **CNNwithMonth:** Inputs only `Temperature`, after passing through Conv layers, concatenate with circularly encoded month feature. As shown in figure (b)

* **LSTMmodel:** Just a simple LSTM model. Inputs only `Temperature`

* **TrivialModel:** This model just outputs the last x value. Just for comparison with other models.

|**(a)CNNsepFeature**|**(b)CNNwithMonth**|
|-------|---------|
|![](./img/sepCNN.PNG)|![](./img/CNNwithMon.PNG)|

For xgb, in `xgboost/`:
* **train.py** to train model. The feature may be raw or been normalized
	
* **train_cross.py** to do cross validation and testing for each year
	
* **train_PCA.py** the feature been PCAed

* **tuning.py** to tune parameters with raw feature or normalized

* **tuning_PCA.py** to tune parameters with PCAed feature

## 5. Results
We use mean-absolute-error (unit: degrees celcius) as a metric to evaluate goodness of a model. Let's see how well they perfrom:

|Model|mae|
|---|---|
|**TrivialModel**|1.92|
|**CNNmodel**|1.398|
|**CNNmodel_v2**|1.402|
|**CNNwithMonth**|1.35|
|**CNNconcatFeature**|1.399|
|**CNNsepFeature**|1.444|
|**LSTM**|1.414|

We tried feeding additional weather feature to improve accuracy, such as pressure and humidity (CNNsep). But turns out past temperature is still the most dominent feature. However, we found that adding month as feature slightly improved our result (CNNwithMonth).

Also, we found that our model performs particularly bad between winter and spring. As the plot below shows, error in the first few months are relatively high.    

![](/img/bigerror.PNG)
  
  
  
  
  
  
  
