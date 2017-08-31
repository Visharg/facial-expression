# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:35:42 2017

@author: Visharg Shah
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
import pickle as cPickle
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dataprocessing

def model_generate():
    img_rows, img_cols = 48, 48
    model = Sequential()
    model.add(Conv2D(64, (5, 5), border_mode='valid',
                            input_shape=(img_rows, img_cols,1)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2)))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
      
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1))) 
    model.add(Conv2D(64, (3, 3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1))) 
    model.add(Conv2D(64,(3, 3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
     
      
    model.add(Dense(7))
      
      
    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model

img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 1200
img_channels = 1

Train_x, Train_y, Val_x, Val_y = dataprocessing.load_data()

Train_x = numpy.asarray(Train_x) 
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols,1)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = model_generate()

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=40,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    horizontal_flip=True,  
    vertical_flip=False) 

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    steps_per_epoch=Train_x.shape[0]/batch_size,
                    epochs=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[checkpointer])