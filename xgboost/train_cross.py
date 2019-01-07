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
#valid = sys.argv[2]
#test = sys.argv[3]
"""
place = {
    'train' : train,
    'validation' : valid,
    'test' : test
}
"""
if train == 'Tainan':
    depth = 7     #T: 7; Ta: 7; Y: 9
    weight = 7    #T: 7; Ta: 5; Y: 8
elif train == 'Yongkang':
    depth = 9     #T: 7; Ta: 7; Y: 9
    weight = 8    #T: 7; Ta: 5; Y: 8
elif train == 'Taoyuan':
    depth = 7     #T: 7; Ta: 7; Y: 9
    weight = 5    #T: 7; Ta: 5; Y: 8
sample = 1

#image_path = './new_image/Tr-{}_Va-{}12_Te-{}10.png'.format(train, valid, test)
#image_path = './Tr-{}_Va-{}12_Te-{}10.png'.format(train, valid, test)
#train_path = './Tainan/2010_1-1_to_2017_12-31.csv'
#test_path = './Tainan/2018_1-1_to_2018_10-31.csv'

#train_path = './Yongkang/2010_1-1_to_2017_12-31.csv'
#test_path = './Yongkang/2018_1-1_to_2018_10-31.csv'

train_path = './{}/2010_1-1_to_2017_12-31.csv'.format(train)
test_path = './{}/2018_1-1_to_2018_10-31.csv'.format(train)

#validation_path = './{}/2010_1-1_to_2017_12-31.csv'.format(valid)

#get data
data_set = get_data.from_2010(train_path, 4, t_temp, t_temp, 24)
data_time = get_data.get_time(train_path, t_temp, 24)
data_set1 = get_data.from_2010(test_path, 4, t_temp, t_temp, 24)
data_time1 = get_data.get_time(test_path, t_temp, 24)

data_set[0] /= 40
data_set[1] /= 40
data_set1[0] /= 40
data_set1[1] /= 40


data_set[0] = pd.concat([data_set[0], data_time[0], data_time[1]], axis=1).reset_index(drop = True)
data_set1[0] = pd.concat([data_set1[0], data_time1[0], data_time1[1]], axis=1).reset_index(drop = True)

all_data = [pd.concat([data_set[0], data_set1[0]], axis=0).reset_index(drop = True), pd.concat([data_set[1], data_set1[1]], axis=0).reset_index(drop = True)]
#all_data[0] = pd.concat([data_set[0], data_set1[0]], axis=0).reset_index(drop = True)
#all_data[1] = pd.concat([data_set[1], data_set1[1]], axis=0).reset_index(drop = True)


training = []
validation = []
testing = []
index = 24*365

#cut validation from train
for i in range(8):
    temp = [all_data[0].iloc[i*index: (i+1)*index, :], all_data[1].iloc[i*index: (i+1)*index, :]]
    validation.append(temp)
    #validation[i][0] = all_data[0].iloc[i*index: (i+1)*index, :]
    #validation[i][1] = all_data[1].iloc[i*index: (i+1)*index, :]
    if i == 7:
        temp = [all_data[0].iloc[(i+1)*index: , :], all_data[1].iloc[(i+1)*index: , :]]
        testing.append(temp)
        #testing[i][0] = all_data[0].iloc[(i+1)*index: , :]
        #testing[i][1] = all_data[1].iloc[(i+1)*index: , :]
    else:
        temp = [all_data[0].iloc[(i+1)*index: (i+2)*index, :], all_data[1].iloc[(i+1)*index: (i+2)*index, :]]
        testing.append(temp)
        #testing[i][0] = all_data[0].iloc[(i+1)*index: (i+2)*index, :]
        #testing[i][1] = all_data[1].iloc[(i+1)*index: (i+2)*index, :]
    if i == 0:
        temp = [all_data[0].iloc[(i+2)*index:, :], all_data[1].iloc[(i+2)*index:, :]]
        training.append(temp)
        #training[i][0] = all_data[0].iloc[(i+2)*index:, :]
        #training[i][1] = all_data[1].iloc[(i+2)*index:, :]
    else:
        temp = [pd.concat([all_data[0].iloc[: i*index, :], all_data[0].iloc[(i+2)*index:, :]], axis=0).reset_index(drop=True), pd.concat([all_data[1].iloc[: i*index, :], all_data[1].iloc[(i+2)*index:, :]], axis=0).reset_index(drop=True)]
        training.append(temp)
        #training[i][0] = pd.concat([all_data[0].iloc[: i*index, :], all_data[0].iloc[(i+2)*index:, :]], axis=0).reset_index(drop=True)
        #training[i][1] = pd.concat([all_data[1].iloc[: i*index, :], all_data[1].iloc[(i+2)*index:, :]], axis=0).reset_index(drop=True)

    
for i in range(8):
    train_x = np.array(training[i][0]).astype(float)
    train_y = np.array(training[i][1]).astype(float)
    dtrain = xgb.DMatrix(train_x, train_y)

    validation_x = np.array(validation[i][0]).astype(float)
    validation_y = np.array(validation[i][1]).astype(float)
    valid = xgb.DMatrix(validation_x, validation_y)

    testing_x = np.array(testing[i][0]).astype(float)
    testing_y = np.array(testing[i][1]).astype(float)
    test = xgb.DMatrix(testing_x)

    params = {
        'nthread' : 4,
        'tree_method' : 'gpu_hist',
        'silent' : 1,
        #'eta' : 0.1,
        'max_depth' : depth,
        'min_child_weight' : weight,
        'subsample' : sample,
    }

    watchlist = [(valid, "test")]

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

    pred = best_model.predict(test)
    guess = np.squeeze(pred*40)
    ans = np.squeeze(testing_y*40)

    mae = 0
    mse = 0
    error = []
    lower = 0
    higher = 0
    for j in range(guess.shape[0]):
        if abs(ans[j] - guess[j]) < 1:
            lower += 1
        elif abs(ans[j] - guess[j]) > 2:
            higher += 1
        error.append(abs(ans[j] - guess[j]))
        mae += abs(ans[j] - guess[j])
        mse += (ans[j] - guess[j])*(ans[j] - guess[j])

    mae = mae / guess.shape[0]
    mse = mse / guess.shape[0]

    #print ("T{}, H{}, P{}".format(t_temp, t_humd, t_pres))
    print (train)
    print ("this is the {} test".format(i))
    #print (params)
    print ("mae : ", mae)
    print ("mse : ", mse)
    print ("higher than 2: ", higher/guess.shape[0]*100)
    print ("lower than 1: ", lower/guess.shape[0]*100)
    plt.plot(error, linewidth = 0.2)
    plt.savefig('./{}_{}.png'.format(train, i))
    plt.clf()
    

print ("cost time: {}\n\n".format(time.time() - now))




