import matplotlib.pyplot as plt
import random
import logging
from keras import layers, models
from keras.metrics import RootMeanSquaredError as rmse
from keras.layers import Dropout
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Input, Embedding
from keras.metrics import RootMeanSquaredError as rmse
from keras.layers import Conv1D, MaxPooling1D, SpatialDropout1D, LSTM, GRU
import timeit
import numpy as np
from keras import initializers
import keras
import tensorflow as tf
import split

# print(split.Return_values())

x_train, x_test, y_train, y_test, Vocab, vector_space, max_length= split.Return_values()


max_epochs = 50
max_acc = 0
# x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Embedding(Vocab, output_dim= vector_space, input_length = max_length))

model.add(GRU(50, activation = 'tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(60, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation = 'tanh'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= 'Adam', metrics=['accuracy'])

model.summary()

# A list for all the accuracies
accuracies = []

# A list of all the calculated times

elapsed_time = []

# A list of all losses

ls = []

# Iterations

iterations = []

# Finally trained weights
trained_weights = []
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

for i in range(max_epochs):
    
    start = timeit.default_timer()
    history = model.fit(x_train, y_train, epochs=50, batch_size = 40, verbose=0)
    stop = timeit.default_timer()
    loss, acc = model.evaluate(x_test, y_test, verbose = 1)
    if(acc > max_acc):
        max_acc = acc
        oldModel = model
    accuracies.append(acc)
    elapsed_time.append(stop - start)
    ls.append(loss)
    iterations.append(i)
    trained_weights.append(model.get_weights())
    
print(max_acc)


plt.figure()
plt.plot(iterations, ls, 'bo-')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('loss.png')

plt.figure()
plt.plot(iterations, elapsed_time, 'bo-')
plt.xlabel('Iterations')
plt.ylabel('Elapsed_time(ms)')
plt.savefig('elapsed.png')

for i in range (len(accuracies)):
    accuracies[i] = 100 * accuracies[i]
    
plt.figure()
plt.plot(iterations, accuracies, 'bo-')
plt.xlabel('Iterations')
plt.ylabel('Accuracy(%)')
plt.savefig('acc.png')




loss, accuracy = model.evaluate(x_test, y_test, verbose = 0)
print('Accuracy: %f' % (accuracy*100))

for i in range(len(accuracies)):
    if(accuracies[i] == max_acc):
        print(trained_weights[i])

history = model.fit(x_train, y_train, epochs=50, batch_size = 40, verbose=1)

measure_time = [18*50,18*50,17*50,17*50,18*50,19*50,20*50,20*50,17*50,16*50,20*50,20*50,18*50,19*50,19*50,19*50,22*50,22*50,20*50,20*50,22*50,18*50,17*50,19*50,18*50,28*50,16*50,15*50,18*50,16*50,16*50,17*50,16*50,16*50,15*50,18*50,17*50,20*50,19*50,16*50,17*50,21*50,19*50,19*50,20*50,17*50,20*50,21*50,19*50,18*50,]

measure_time

epoch = []

for(i) in range(50):

    epoch.append(i)

plt.figure()
plt.plot(epoch, measure_time, 'bo-')
plt.xlabel('Epoch')
plt.ylabel('Time (ms)')
plt.savefig('measure_time.png')

print(history.history['loss'])

epoch = []
loss = []

for i in range(len(history.history['loss'])):
    epoch.append(i)
    loss.append(history.history['loss'][i])


plt.figure()
plt.plot(epoch, loss, 'bo-')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.savefig('Training_Loss.png')
	

