from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler


stock = pd.read_csv('new_python_stock.csv')
features = list(stock.columns)
target_var = 'y'
name = 'name'
features.remove(target_var)
features.remove(name)

X = stock.iloc[:, 2:-1]
zdf = stock.iloc[:, -1] + stock.iloc[:, 'name']
############
# plt.hist(y, 40, normed=1, histtype='stepfilled', facecolor='b', alpha=0.75)
# plt.title('Histogram')
# plt.show()
#########

# print(sum(y > 10), sum(y <= 10))
train_x, test_x, train_y_zdf, test_y_zdf = train_test_split(X,
                                                    zdf,
                                                    train_size=0.8,
                                                    test_size=0.2, random_state=0)
'''
train_y_zdf[train_y_zdf>=10] = 2
train_y_zdf[train_y_zdf<10 & train_y_zdf>0] = 1
train_y_zdf[train_y_zdf<=0] = 0
'''

train_y = np.where(train_y_zdf > 15, 1, 0)
test_y = np.where(test_y_zdf > 15, 1, 0)
# train_x, test_x, train_y, test_y
# train_x.head(), test_x.head(), train_y.head(), test_y.head()

sc = StandardScaler()
X_train = sc.fit_transform(train_x)
X_test = sc.transform(test_x)

classifier = Sequential()
classifier.add(Dense(units=512, kernel_initializer='uniform', activation='relu', input_dim=X.shape[1]))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='tanh'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
classifier.fit(X_train, train_y, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

'''stock['y_pred'] = np.NaN
stock.iloc[(len(stock) - len(y_pred)):,-1:] = y_pred
trade_dataset = stock.dropna()
'''

eval = classifier.evaluate(test_x, test_y, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
      % (eval[0], eval[1] * 100))

test = test_x
test['Predict y'] = y_pred
test['True y zdf'] = test_y_zdf

final = test[test['Predict y'] == 1]

final
sum(final.iloc[:, -1] > 0)
sum(final.iloc[:, -1] > 0)/final.shape[0]
final.index
stock2 = np.array(stock)
select = stock2[final.index]

final_df = pd.DataFrame(select[:,1])
final_df['涨幅'] = select[:, -2]


nan_excel = pd.DataFrame()
nan_excel.to_excel('Save_Excel.xlsx')
writer = pd.ExcelWriter('Save_Excel.xlsx')
final_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
writer.save()