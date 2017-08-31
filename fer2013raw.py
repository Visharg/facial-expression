# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:42:10 2017

@author: Visharg Shah
"""

import numpy as np
import pandas as pd


def faces_load_data():
 
    skip_rows = 1
    train_size = 28709
    test_size = 3589
    dim = 48
    X_train = np.empty([train_size,dim, dim])
    X_test = np.empty([test_size, dim, dim])
    y_train = np.empty(train_size)
    y_test = np.empty(test_size)
    
    f = open('fer2013.csv', 'r')
 
    train_index = test_index = 0
    for i, line in enumerate(f):
        if i >= skip_rows:
            split_line = line.split(",")
            usage = split_line[2].rstrip()
            if usage == 'Training':
                X_train[train_index, :,:] = np.fromstring(split_line[1], dtype = 'int', sep = ' ').reshape(dim, dim)
                y_train[train_index] = int(split_line[0])
                train_index += 1
            elif usage == 'PublicTest':
                X_test[test_index, :,:] = np.fromstring(split_line[1], dtype = 'int', sep = ' ').reshape(dim, dim)
                y_test[test_index] = int(split_line[0])
                test_index += 1
                 
    return (X_train, y_train) , (X_test, y_test)



np.random.seed(1337) 
    
(X_train, y_train), (X_test, y_test) = faces_load_data()

print (X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 48,48,1)
X_test = X_test.reshape(X_test.shape[0], 48,48,1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train, 7)
Y_test = np_utils.to_categorical(y_test, 7)


Y_train_1 = pd.DataFrame(Y_train)
Y_train_1[0] = Y_train_1[0] + Y_train_1[1]
Y_train_1 = Y_train_1.drop(1, 1)
Y_train_1 = Y_train_1.as_matrix(columns = Y_train_1.columns[:])


Y_test_1 = pd.DataFrame(Y_test)
Y_test_1[0] = Y_test_1[0] + Y_test_1[1]
Y_test_1 = Y_test_1.drop(1, 1)
Y_test_1 = Y_test_1.as_matrix(columns = Y_test_1.columns[:])


#########

###########
############

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adadelta, Adam

#####

facial = Sequential()

#Input Layer
facial.add(Convolution2D(32, (5,5), input_shape = (48,48,1), padding='same', activation='relu'))
facial.add(Convolution2D(32, (5,5), padding = 'same',activation='relu'))
facial.add(Convolution2D(32, (5,5), padding = 'same',activation='relu'))

            
facial.add(MaxPooling2D(pool_size=(2,2)))

facial.add(Convolution2D(64, (5,5), input_shape = (48,48,1), padding='same', activation='relu'))
facial.add(Convolution2D(64, (5,5), padding='same', activation='relu'))
facial.add(Convolution2D(64, (5,5), padding = 'same',activation='relu'))


            
facial.add(MaxPooling2D(pool_size=(2,2)))

facial.add(Convolution2D(128, (5,5), input_shape = (48,48,1), padding='same', activation='relu'))
facial.add(Convolution2D(128, (5,5), padding='same', activation='relu'))
facial.add(Convolution2D(128, (5,5), padding = 'same',activation='relu'))


            
facial.add(MaxPooling2D(pool_size=(2,2)))

facial.add(Flatten())
facial.add(Dense(128, activation = 'relu'))
facial.add(Dense(64, activation = 'relu'))
facial.add(Dropout(0.5))
facial.add(Dense(7, activation='softmax'))
facial.add(Dropout(0.5))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
facial.compile(loss='categorical_crossentropy',optimizer=ada,metrics=['accuracy'])
#facial.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

facial.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=1, validation_data=(X_test, Y_test))
score = facial.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])