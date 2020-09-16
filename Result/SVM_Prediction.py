import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
import backtrader as bt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
###########
# Buy
stock_data = pd.read_csv('git_project/git_project/stock/kcb.csv')

stock_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
stock_code = np.array(stock_data['WINDCODE'].drop_duplicates())
names = locals()
for code in stock_code:
    names['df' + str(code)] = stock_data[stock_data['WINDCODE'] == code]

###
start_date = '2019-08-28'
end_date = '2020-04-13'

test_code = []
for i in range(len(stock_code)):
    df = names['df' + str(stock_code[i])]
    date = df['date'].tolist()
    if start_date in date:
        test_code.append(stock_code[i])
        start_index, end_index = date.index(start_date), date.index(end_date)
        names['train_data' + str(stock_code[i])] = df.iloc[start_index:end_index]
        names['test_data' + str(stock_code[i])] = df.iloc[end_index:]
        names['whole_data'+ str(stock_code[i])] = df.iloc[start_index:]

###########
# Preparation
whole_data = names['whole_data'+ str(stock_code[0])]

var = ['CLOSE', 'VOLUME', 'HIGH', 'LOW']
names = locals()
for i in range(len(var)): names['list' + str(var[i])] = []
vol = []
return_x = []
qjstd = []

for i in range(len(whole_data) - 14):
    vol_data = np.array(whole_data['VOLUME'])
    vol.append(vol_data[i + 14])
    close_data = np.array(whole_data['CLOSE'])
    return_x.append(close_data[i + 14]/close_data[i])
    qjstd.append(np.std(close_data[i:i + 15]))
    for ind in range(len(var)):
        array = np.array(whole_data[var[ind]])
        data = array[i + 14] / np.mean(array[i:i + 15])
        names['list' + str(var[ind])].append(data)

x_all = pd.DataFrame({'close_mean': names['list' + str(var[0])],
                     'volume_mean': names['list' + str(var[1])],
                     'high_mean': names['list' + str(var[2])],
                     'low_mean': names['list' + str(var[3])],
                      'vol' : vol, 'return_x' : return_x, 'std' : qjstd})


y_total = names['whole_data'+ str(stock_code[0])][['date','CLOSE']]
y_total_close = np.array(y_total['CLOSE'])
y_all = []
for i in range(len(y_total_close) - 20):
    if y_total_close[i + 20] > y_total_close[i + 14]:
        label = 1
    else:
        label = 0
    y_all.append(label)

###
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(data.shape[0], 1)
data = scaler.fit_transform(data)




###########
# Train
train_range = 100

x_train = x_all.loc[:train_range-1]
y_train = y_all[:train_range]
svm_model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                      tol=0.001, cache_size=200, verbose=False, max_iter=-1,
                      decision_function_shape='ovr', random_state=None)
svm_model.fit(x_train, y_train)

########################
# Test
y_test = y_all[train_range:]
x_test = x_all.loc[train_range:train_range+len(y_test)-1]

svm_model.predict(x_test)
