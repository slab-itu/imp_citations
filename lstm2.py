# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 05:22:35 2017

@author: mubas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 05:17:58 2017

@author: mubas
"""



from pandas import read_csv

#import matplotlib.pyplot as plt
#import numpy
#import pandas
from keras.models import Sequential
from keras.layers import Activation
#from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from keras import optimizers
import random
from sklearn.metrics import classification_report
random.seed( 3 )


for number in range(100):
    
    # load dataset
    dataset = read_csv('comp.csv', header=None, index_col=None)
    from sklearn.utils import shuffle
    df = dataset
    df = shuffle(dataset)
    values = df.values
    # specify columns to plot
    #groups = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
    #i = 1
    # plot each column
    #pyplot.figure()
    #for group in groups:
    #	pyplot.subplot(len(groups), 1, i)
    #	pyplot.plot(values[:, group])
    #	pyplot.title(dataset.columns[group], y=0.5, loc='right')
    #	i += 1
    #pyplot.show()
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    train = values[:289, :]
    test = values[289:, :]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(52,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Activation('softmax'))
    model.add(LSTM(26,return_sequences=True))
    model.add(Activation('softmax'))
    model.add(LSTM(13,return_sequences=True))
    model.add(Activation('softmax'))
    model.add(LSTM(7,return_sequences=True))
    model.add(Activation('softmax'))
    model.add(LSTM(3,return_sequences=True))
	model.add(LSTM(1))
    '''
    model.add(Activation('softmax'))
    model.add(LSTM(1))
    #model.add(Activation('softmax'))
    #model.add(Dense(1))
    #sgd = optimizers.SGD(lr=1.0, decay=1e-6, momentum=0.1, nesterov=True)
	'''
    model.compile(loss='mean_squared_error', optimizer="adam",metrics=['accuracy'])
    # fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    
    scores = model.evaluate(train_X, train_y)
    
    # print the classification report
    
    predicted = model.predict_classes(test_X)
    report = classification_report(test_y, predicted)
    print('\n')
    
    print(report)
    print(scores)
    import csv
    with open("result.txt", "a", encoding = 'iso-8859-1"',newline='') as file:
                    writer = csv.writer(file, delimiter = ",")
                    writer.writerow([report])
                    writer.writerow(scores)
                
                




# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = numpy.concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = numpy.concatenate((test_y, test_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = numpy.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
