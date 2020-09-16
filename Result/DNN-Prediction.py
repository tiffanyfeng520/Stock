import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import RMSprop
from keras import


stock = pd.read_csv('python_stock.csv')
features = list(stock.columns)
target_var = 'y'
name = 'name'
code = 'code'
features.remove(target_var)
features.remove(name)
features.remove(code)
features

scaler = MinMaxScaler(feature_range=(0, 1))
stock[features] = scaler.fit_transform(stock[features])
y = np.array(stock[target_var])
y = y.reshape(y.shape[0],1)
y = scaler.fit_transform(y)
train_x, test_x, train_y, test_y = train_test_split(stock[features],
                                                    y,
                                                    train_size=0.8,
                                                    test_size=0.2, random_state=0)
train_x, test_x, train_y, test_y
train_x.head(), test_x.head(), train_y.head(), test_y.head()


####################
model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(6,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss="mean_squared_error",optimizer=RMSprop(),metrics=['accuracy'])
model.fit(train_x, train_y, epochs=50, batch_size=128)
model.save('dnn.h5')

#######
y_predict = model.predict(test_x)
inv_y_predict = scaler.inverse_transform(y_predict)
inv_y_true = scaler.inverse_transform(test_y)
MSE = mean_squared_error(inv_y_predict, inv_y_true)
print(MSE)


plt.figure(figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(inv_y_predict,color="b",label="predict")
plt.plot(inv_y_true,color="r",label="true")
plt.show()