import xgboost as xgb
import numpy as np
import pandas as pd
import time
import sys
from sklearn.decomposition import PCA

import get_data

now = time.time()

#train_path = sys.argv[1]
#test_path = sys.argv[2]
components = int(sys.argv[1])

t_temp = 96
t_humd = 96#int(sys.argv[2])
t_pres = 0#int(sys.argv[3])
#get data
data_set = get_data.from_2010('./Taoyuan/2010_1-1_to_2017_12-31.csv', 4, t_temp, t_temp, 24)
data_set[0] /= 40
data_time = get_data.get_time('./Taoyuan/2010_1-1_to_2017_12-31.csv', t_temp, 24)
#data_set = get_data.from_2010(train_path, 4, t_temp, t_temp, 24)
#data_time = get_data.get_time(train_path, t_temp, 24)
#data_humd = get_data.from_2010('./Tainan/2010_1-1_to_2017_12-31.csv', 6, t_temp, t_temp, 24)
#data_set[0] = pd.concat([data_set[0], data_humd[0]], axis=1).reset_index(drop = True)

pca_t1 = PCA(n_components = 6)
pca_t2 = PCA(n_components = 6)
pca_t3 = PCA(n_components = 12)
#pca_t4 = PCA(n_components = 12)
#pca_h = PCA(n_components = 24)
aPCA_temp1 = pd.DataFrame.from_records(pca_t1.fit_transform(data_set[0].iloc[:, : 48]))
aPCA_temp2 = pd.DataFrame.from_records(pca_t2.fit_transform(data_set[0].iloc[:, 48: 72]))
aPCA_temp3 = pd.DataFrame.from_records(pca_t3.fit_transform(data_set[0].iloc[:, 72: ]))
#aPCA_temp4 = pd.DataFrame.from_records(pca_t4.fit_transform(data_set[0].iloc[:, 72: ]))
#aPCA_humd = pd.DataFrame.from_records(pca_h.fit_transform(data_humd[0]))
#data_set[0] /= 40
#data_set[1] /= 40

data_set[0] = pd.concat([aPCA_temp1, aPCA_temp2, aPCA_temp3, data_time[0], data_time[1]], axis=1).reset_index(drop = True)
#data_set[0] = pd.concat([data_set[0], data_time[0], data_time[1]], axis=1).reset_index(drop = True)
#get test data
test_set = get_data.from_2010('./Taoyuan/2018_1-1_to_2018_10-31.csv', 4, t_temp, t_temp, 24)#####
test_set[0] /= 40
test_time = get_data.get_time('./Taoyuan/2018_1-1_to_2018_10-31.csv', t_temp, 24)
#test_set = get_data.from_2010(test_path, 4, t_temp, t_temp, 24)#####
#test_time = get_data.get_time(test_path, t_temp, 24)
#test_humd = get_data.from_2010('./Tainan/2018_1-1_to_2018_10-31.csv', 6, t_temp, t_temp, 24)
#pca = PCA(n_components = 24)

#test_set[0] = pd.concat([test_set[0], test_humd[0]], axis=1).reset_index(drop = True)
aPCA_temp1 = pd.DataFrame.from_records(pca_t1.transform(test_set[0].iloc[:, :48]))
aPCA_temp2 = pd.DataFrame.from_records(pca_t2.transform(test_set[0].iloc[:, 48: 72]))
aPCA_temp3 = pd.DataFrame.from_records(pca_t3.transform(test_set[0].iloc[:, 72: ]))
#aPCA_temp4 = pd.DataFrame.from_records(pca_t4.transform(test_set[0].iloc[:, 72: ]))
#aPCA_humd = pd.DataFrame.from_records(pca_h.transform(test_humd[0]))


#test_set[0] /= 40
#test_set[1] /= 40

test_set[0] = pd.concat([aPCA_temp1, aPCA_temp2, aPCA_temp3, test_time[0], test_time[1]], axis=1).reset_index(drop = True)
#test_set[0] = pd.concat([test_set[0], test_time[0], test_time[1]], axis=1).reset_index(drop = True)

#cut validation from train
train_end = 64225                   #2017-1-1 = 64225
validation_start = 64225            #2017-11-1 = 71521
validation_end = -1            #2017-5-1 = 67105
testing_start = 0              #2018-5-1 = 2881
testing_end = -1
#train
train_data = np.array(data_set[0].iloc[: train_end, :]).astype(float)
label_data = np.array(data_set[1].iloc[: train_end, :]).astype(float)
dtrain = xgb.DMatrix(train_data, label_data)
#validation
valid_x = np.array(data_set[0].iloc[validation_start: validation_end, :]).astype(float)
valid_y = np.array(data_set[1].iloc[validation_start: validation_end, :]).astype(float)
validation = xgb.DMatrix(valid_x, valid_y)
#testing
testing_x = np.array(test_set[0].iloc[testing_start: testing_end, :]).astype(float)
testing_y = np.array(test_set[1].iloc[testing_start: testing_end, :]).astype(float)
testing = xgb.DMatrix(testing_x)

#start tuning
boost_round = 5000
early_stopping = 30

params = {
    'nthread' : 2,
    'tree_method' : 'gpu_hist',
    'silent' : 1,

}
best_params = None

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(4,7)
    for min_child_weight in range(2, 10)
]

min_rmse = float('inf')
for max_depth, min_child_weight in gridsearch_params:
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round = boost_round,
        seed = 42,
        nfold = 7,
        metrics = {'rmse'},
        early_stopping_rounds = early_stopping
    )

    mean_rmse = cv_result['test-rmse-mean'].min()
    rounds = cv_result['test-rmse-mean'].idxmin()
    print ("\tRMSE {} for {} rounds".format(mean_rmse, rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth, min_child_weight)

#print ("\nBest params: {}, {}, RMSE: {}\n".format(best_params[0], best_params[1], min_rmse))
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]


gridsearch_params = [
    subsample
    for subsample in [i/50. for i in range(45, 51)]
]
best_param = None

min_rmse = float('inf')


for subsample in gridsearch_params:
    #print ("CV with subsample = {}".format(subsample))

    params['subsample'] = subsample

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round = boost_round,
        seed = 42,
        nfold = 7,
        metrics = {'rmse'},
        early_stopping_rounds = early_stopping
    )

    mean_rmse = cv_result['test-rmse-mean'].min()
    rounds = cv_result['test-rmse-mean'].idxmin()
    #print ("\tRMSE {} for {} rounds".format(mean_rmse, rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_param = subsample

#print ("\nBest param: {}, RMSE: {}\n".format(best_param, min_rmse))
params['subsample'] = best_param

print (params)

watchlist = [(validation, "test")]

model = xgb.train(
    params,
    dtrain,
    boost_round,
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
#guess = np.squeeze(pred*40)
#ans = np.squeeze(testing_y*40)
guess = np.squeeze(pred)
ans = np.squeeze(testing_y)


mae = 0
mse = 0
for i in range(guess.shape[0]):
    mae += abs(ans[i] - guess[i])
    mse += (ans[i] - guess[i])*(ans[i] - guess[i])

mae = mae / guess.shape[0]
mse = mse / guess.shape[0]

print ("T{}, H{}, P{}".format(t_temp, t_humd, t_pres))
print (params)
print ("mae : ", mae)
print ("mse : ", mse)

print ("cost time: {}\n\n".format(time.time() - now))


