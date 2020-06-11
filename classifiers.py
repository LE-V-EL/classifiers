################################################################
# This file contains the classifiers for LE-V-EL platform.
# AlexNet
# DenseNet
# VGG
################################################################
import pickle

from keras import models
from keras import layers
from keras import optimizers
import keras.applications
import keras.callbacks
from keras import backend as K
from keras.utils.np_utils import to_categorical
import sklearn.metrics
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import sys
import time
from datetime import datetime


class classifiers:
    def __init__(self):
        pass


    def createKerasClassifiers(CLASSIFIER,X_train,X_val,X_test):
            X_train_3D = np.stack((X_train,) * 3, -1)
            X_val_3D = np.stack((X_val,) * 3, -1)
            X_test_3D = np.stack((X_test,) * 3, -1)

            if CLASSIFIER == 'VGG19':
                feature_generator = keras.applications.VGG19(weights=None, include_top=False, input_shape=(100, 100, 3))
            elif CLASSIFIER == 'DenseNet121':
                feature_generator = keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(100, 100, 3))
            return feature_generator,X_train_3D,X_val_3D,X_test_3D


    def createScratchClassifiers(CLASSIFIER,X_train,X_val,X_test,optimizer,loss):
            X_train_3D = np.stack((X_train,) * 3, -1)
            X_val_3D = np.stack((X_val,) * 3, -1)
            X_test_3D = np.stack((X_test,) * 3, -1)

            if CLASSIFIER == 'ALEXNET':
                model = Sequential()
                model.add(
                    Conv2D(filters=96, input_shape=(100, 100, 1), kernel_size=(11, 11), strides=(4, 4), padding="valid",
                           activation="relu"))
                model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
                model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
                model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
                model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
                model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
                model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
                model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

                # Normalize the ALexNet classifier

                model.add(Flatten())
                model.add(layers.Dense(256, activation='relu', input_dim=(100, 100, 3)))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(1, activation='linear'))
                if optimizer == "sgd":
                    opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

                elif optimizer == "ADAM":
                    opt = keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name="Adam")

                elif optimizer == "RMSPROP":
                    opt = keras.optimizers.RMSprop( learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")

                elif optimizer == "ADAMAX":
                    opt = keras.optimizers.Adamax( learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")

                if loss == "MSEMAE":
                    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])  # MSE for regression

                if loss == "categorical_crossentropy":
                    model.compile(loss=keras.losses.categorical_crossentropy(), optimizer=opt, metrics=['mse', 'mae'])

                if loss == "binary_crossentropy":
                    model.compile(loss=keras.losses.binary_crossentropy(), optimizer=opt, metrics=['mse', 'mae'])

                if loss == "KLD":
                    model.compile(loss=keras.losses.kld(), optimizer=opt, metrics=['mse', 'mae'])


            return model,X_train_3D,X_val_3D,X_test_3D

    def addOptimizerLoss(self):
        return 0



    def normaliseClassifiers(feature_generator,optimizer,loss):
        MLP = models.Sequential()
        MLP.add(layers.Flatten(input_shape=feature_generator.output_shape[1:]))
        MLP.add(layers.Dense(256, activation='relu', input_dim=(100, 100, 3)))
        MLP.add(layers.Dropout(0.5))
        MLP.add(layers.Dense(1, activation='linear'))  # REGRESSION

        model = keras.Model(inputs=feature_generator.input, outputs=MLP(feature_generator.output))


        return model


    def trainClassifer(model, CLASSIFIER, X_train_3D, y_train,X_val_3D,y_val,epochs,batch_size,MODELFILE):
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),keras.callbacks.ModelCheckpoint(MODELFILE, monitor='val_loss', verbose=1, save_best_only=True,mode='min')]

        history = model.fit(X_train_3D, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_3D, y_val),
                            callbacks=callbacks, verbose=True)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        file_pi = "history" +  CLASSIFIER + str(dt_string)
        with open(MODELFILE, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

