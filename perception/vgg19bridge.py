import os, sys, time

import keras
from keras.models import load_model

import pickle as p
from keras import layers
from keras import optimizers
import keras.applications
import keras.callbacks
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

class VGG19Bridge:

    def __init__(self, model_dir=None, network='VGG19', file_name="weights.h5"):
        '''
        '''
        self.test_model = None
        self.model      = None
        self.network    = network
        self.file_name  = file_name

        if model_dir is not None:

            self.model_dir = model_dir

            if self.network == 'VGG19':
                self.model_dir = os.path.join(self.model_dir, "vgg19")
            elif self.network == 'RESNET':
                self.model_dir = os.path.join(self.model_dir, "resnet")
            elif self.network == 'DENSENET':
                self.model_dir = os.path.join(self.model_dir, "densenet")
            elif self.network == 'ALEXNET':
                self.model_dir = os.path.join(self.model_dir, "alexnet")

            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        else:
            self.model_dir = os.path.dirname(__file__)



    def train(self, x_train, y_train, x_val, y_val, epochs):

        t0 = time.time()

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.model_file = os.path.join(self.model_dir, self.file_name)

        print ('Storing in ', self.model_dir)

        if self.network == 'VGG19':
            feature_generator = keras.applications.VGG19(weights=None, include_top=False, input_shape=(100,100,3))
        elif self.network == 'RESNET':
            feature_generator = keras.applications.ResNet50(weights=None, include_top=False, input_shape=(100,100,3))
        elif self.network == 'DENSENET':
            feature_generator = keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(100,100,3))
        elif self.network == 'ALEXNET':
            alexnet_model = Sequential()
            alexnet_model.add(
                Conv2D(filters=96, input_shape=(100, 100, 3), kernel_size=(11, 11), strides=(4, 4), padding="valid",
                       activation="relu"))
            alexnet_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
            alexnet_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
            alexnet_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
            alexnet_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
            alexnet_model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
            alexnet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
            alexnet_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

            # Normalize the ALexNet classifier

            alexnet_model.add(Flatten())
            alexnet_model.add(layers.Dense(256, activation='relu', input_dim=(100, 100, 3)))
            alexnet_model.add(layers.Dropout(0.5))
            alexnet_model.add(layers.Dense(1, activation='linear'))
            sgd_opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            alexnet_model.compile(loss='mean_squared_error', optimizer=sgd_opt, metrics=['mse', 'mae'])

            print('Alexnet Setup complete after', time.time() - t0)

            t0 = time.time()

            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'), \
                keras.callbacks.ModelCheckpoint(self.model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                                mode='min')]

            history = alexnet_model.fit(x_train,
                                     y_train,
                                     epochs=epochs,
                                     batch_size=32,
                                     validation_data=(x_val, y_val),
                                     callbacks=callbacks,
                                     verbose=True)

            fit_time = time.time() - t0

            p.dump(history.history, open(os.path.join(self.model_dir, "history.p"), "wb"))

            print(self.network, ' Fitting done', time.time() - t0)

            return history


        MLP = keras.models.Sequential()
        MLP.add(keras.layers.Flatten(input_shape=feature_generator.output_shape[1:]))
        MLP.add(keras.layers.Dense(256, activation='relu', input_dim=(100,100,3)))
        MLP.add(keras.layers.Dropout(0.5))
        MLP.add(keras.layers.Dense(1, activation='linear')) # REGRESSION

        self.model = keras.Model(inputs=feature_generator.input, outputs=MLP(feature_generator.output))

        sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse', 'mae']) # MSE for regression

        print (self.network, 'Setup complete after', time.time()-t0)

        t0 = time.time()

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'), \
                     keras.callbacks.ModelCheckpoint(self.model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=32,
                                 validation_data=(x_val, y_val),
                                 callbacks=callbacks,
                                 verbose=True)

        fit_time = time.time()-t0

        p.dump(history.history, open(os.path.join(self.model_dir, "history.p"), "wb"))

        print(self.network, ' Fitting done', time.time()-t0)

        return history



    def predict(self, segmented_images, verbose=False):
        '''
        Predicts a maskr-cnn results dict using VGG19.
        '''

        if not self.test_model:
            model_path = os.path.join(self.model_dir, self.file_name)
            self.test_model = load_model(model_path)

        all_preds = []

        for image in segmented_images:

            vgg_scores = []

            for image_segment in image:
                # predict
                vgg_scores.append(self.test_model.predict([[image_segment]])[0])

            if verbose:
                print(image_segment[0])

            y_image_pred = []

            for v in vgg_scores:
                y_image_pred.append(v[0])

            all_preds.append(y_image_pred)


        return all_preds
