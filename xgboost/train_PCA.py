import xgboost as xgb
import numpy as np
import pandas as pd
import time
import get_data
import seaborn as sns
import matplotlib.pyplot as plt
import sys


    
now = time.time()

t_temp = 96
t_humd = 0
t_pres = 0

train = sys.argv[1]
valid = sys.argv[2]
test = sys.argv[3]

place = {
    'train' : train,
    'validation' : valid,
    'test' : test
}

if train == 'Tainan':
    depth = 7     #T: 7; Ta: 7; Y: 9
    weight = 7    #T: 7; Ta: 5; Y: 8
elif train == 'Yongkang':
    depth = 9     #T: 7; Ta: 7; Y: 9
    weight = 8    #T: 7; Ta: 5; Y: 8
elif train == 'Taoyuan':
    depth = 7     #T: 7; Ta: 7; Y: 9
    weight = 5    #T: 7; Ta: 5; Y: 8
#depth = 8
#weight = 7
sample = 0.98

image_path = './new_image/Tr-{}_Va-{}12_Te-{}10.png'.format(train, valid, test)
#train_path = './Tainan/2010_1-1_to_2017_12-31.csv'
#test_path = './Tainan/2018_1-1_to_2018_10-31.csv'

#train_path = './Yongkang/2010_1-1_to_2017_12-31.csv'
#test_path = './Yongkang/2018_1-1_to_2018_10-31.csv'

train_path = './{}/2010_1-1_to_2017_12-31.csv'.format(train)
test_path = './{}/2018_1-1_to_2018_10-31.csv'.format(test)

validation_path = './{}/2010_1-1_to_2017_12-31.csv'.format(valid)

#get data
data_set = get_data.from_2010(train_path, 4, t_temp, t_temp, 24)
data_time = get_data.get_time(train_path, t_temp, 24)

data_set[0] /= 40
data_set[1] /= 40

data_set[0] = pd.concat([data_set[0], data_time[0], data_time[1]], axis=1).reset_index(drop = True)

#get validation data
data_set1 = get_data.from_2010(validation_path, 4, t_temp, t_temp, 24)
data_time1 = get_data.get_time(validation_path, t_temp, 24)

data_set1[0] /= 40
data_set1[1] /= 40

data_set1[0] = pd.concat([data_set1[0], data_time1[0], data_time1[1]], axis=1).reset_index(drop = True)


#get test data
test_set = get_data.from_2010(test_path, 4, t_temp, t_temp, 24)#####
test_time = get_data.get_time(test_path, t_temp, 24)######

test_set[0] /= 40
test_set[1] /= 40

test_set[0] = pd.concat([test_set[0], test_time[0], test_time[1]], axis=1).reset_index(drop = True)

#cut validation from train
train_end = 64225                   #2017-1-1 = 64225
validation_start = 64225           #2017-11-1 = 71521
validation_end = -1            #2017-5-1 = 67105
testing_start = 0              #2018-5-1 = 2881
testing_end = -1
#train
train_data = np.array(data_set[0].iloc[: train_end, :]).astype(float)
label_data = np.array(data_set[1].iloc[: train_end, :]).astype(float)
dtrain = xgb.DMatrix(train_data, label_data)
#validation
valid_x = np.array(data_set1[0].iloc[validation_start: validation_end, :]).astype(float)
valid_y = np.array(data_set1[1].iloc[validation_start: validation_end, :]).astype(float)
validation = xgb.DMatrix(valid_x, valid_y)
#testing
testing_x = np.array(test_set[0].iloc[testing_start: testing_end, :]).astype(float)
testing_y = np.array(test_set[1].iloc[testing_start: testing_end, :]).astype(float)
testing = xgb.DMatrix(testing_x)

params = {
    'nthread' : 4,
    'tree_method' : 'gpu_hist',
    'silent' : 1,
    #'eta' : 0.1,
    'max_depth' : depth,
    'min_child_weight' : weight,
    'subsample' : sample,
}

watchlist = [(validation, "test")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round = 5000,
    evals = watchlist,
    early_stopping_rounds = 30
)

best_round = model.best_iteration + 1

best_model = xgb.train(
    params,
    dtrain,
    best_round,
    evals = watchlist
)

pred = best_model.predict(testing)
guess = np.squeeze(pred*40)
ans = np.squeeze(testing_y*40)

mae = 0
mse = 0
error = []
for i in range(guess.shape[0]):
    error.append(abs(ans[i] - guess[i]))
    mae += abs(ans[i] - guess[i])
    mse += (ans[i] - guess[i])*(ans[i] - guess[i])

mae = mae / guess.shape[0]
mse = mse / guess.shape[0]

#print ("T{}, H{}, P{}".format(t_temp, t_humd, t_pres))
print (place)
#print (params)
print ("mae : ", mae)
print ("mse : ", mse)

plt.plot(error, linewidth = 0.2)
#plt.savefig(image_path)

print ("cost time: {}\n\n".format(time.time() - now))




