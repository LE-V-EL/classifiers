import os, sys, time

import keras
from keras.models import load_model

import numpy  as np
import pickle as p

class VGG19Bridge:

    def __init__(self, model_dir=None, classifier='VGG19', file_name="vgg19weights.h5"):
        '''
        '''
        t0 = time.time()

        self.test_model = None

        if model_dir is not None:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(os.path.dirname(__file__))

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.model_file = os.path.join(model_dir, file_name)

        print ('Storing in ', self.model_dir)

        if classifier == 'VGG19' or classifier == 'XCEPTION':

            if classifier == 'VGG19':
                feature_generator = keras.applications.VGG19(weights=None, include_top=False, input_shape=(100,100,3))
            elif classifier == 'XCEPTION':
                feature_generator = keras.applications.Xception(weights=None, include_top=False, input_shape=(100,100,3))

            MLP = keras.models.Sequential()
            MLP.add(keras.layers.Flatten(input_shape=feature_generator.output_shape[1:]))
            MLP.add(keras.layers.Dense(256, activation='relu', input_dim=(100,100,3)))
            MLP.add(keras.layers.Dropout(0.5))
            MLP.add(keras.layers.Dense(1, activation='linear')) # REGRESSION

            self.model = keras.Model(inputs=feature_generator.input, outputs=MLP(feature_generator.output))

            sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

            self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse', 'mae']) # MSE for regression

        print ('VGG19 Setup complete after', time.time()-t0)



    def train(self, x_train, y_train, x_val, y_val, epochs):
        '''
        '''
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

        print('VGG19 Fitting done', time.time()-t0)

        return history



    def predict(self, segmented_images, verbose=False, model_path=None):
        '''
        Predicts a maskr-cnn results dict using VGG19.
        '''
        t0 = time.time()

        if model_path is None:
            model_path = self.model_file

        if not self.test_model:
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
                y_image_pred.append(v[0]*90)

            all_preds.append(y_image_pred)

        print('VGG19 Prediction complete after', time.time()-t0)

        return all_preds
