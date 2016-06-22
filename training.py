"""
(C) 2016 Hazem Nomer
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time
import sys

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, TimeDistributed
from keras.callbacks import Callback
from processing import *
TRAIN_SIZE = 21
TARGET_TIME = 1
HIDDEN_RNN =500
LAG_SIZE = 1
EMB_SIZE = 1

print 'Data loading...'  
timeseries, dates = load_snp_close()
dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
#plt.plot(dates, timeseries)

X, Y = split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
X,Y = createnumpyarray(X,Y,TRAIN_SIZE)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, percentage=0.9)

Xp, Yp = split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
Xp, Yp = createnumpyarray(Xp,Yp,TRAIN_SIZE)
X_trainp, X_testp, Y_trainp, Y_testp = create_Xt_Yt(Xp, Yp, percentage=0.9)


print 'Building model...'
model = Sequential()
model.add(LSTM(input_length = 1, input_dim=TRAIN_SIZE, output_dim=HIDDEN_RNN, return_sequences=True))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(1)))
model.add(Activation('linear'))
model.compile(optimizer='adam', 
              loss='mse')

model.fit(X_train, 
          Y_train, 
          nb_epoch=5,
          batch_size = 32,
          verbose=1, 
          validation_split=0.3)
score = model.evaluate(X_test, Y_test, batch_size=32)
print score


params = []
for xt in X_testp:
    xt = np.array(xt)
    mean_ = xt.mean()
    scale_ = xt.std()
    params.append([mean_, scale_])

predicted = model.predict(X_test)
new_predicted = []
predicted= predicted.reshape(1669,1,1,1).swapaxes(1,2).reshape(1*1669,-1)
for pred, par in zip(predicted, params):
    a = pred*par[1]
    a += par[0]
    new_predicted.append(a)


mse = mean_squared_error(predicted, new_predicted)
print mse
Y_test=Y_test[:,0,:]
Y_testp= Y_testp[:,0,:]
fig = plt.figure()
#plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
#plt.plot(predicted[:150], color='blue') # RED - trained PREDICTION
plt.plot(Y_testp[:150], color='green') # GREEN - actual RESULT
plt.plot(new_predicted[:150], color='red') # ORANGE - restored PREDICTION
plt.title('trained result vs prediction')
plt.legend(['Actual','Predictied'], loc=(0, 0))
plt.show()