#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:21:55 2019

@author: vladgriguta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:45:57 2019

@author: vladgriguta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

occurancePercentage = 0.05

locationPreprocSpectra = 'preprocessedData400K/'
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import confusion_matrix #, classification_report, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.initializers import random_uniform
from keras import layers
from keras.optimizers import RMSprop
import gc
gc.collect()

# load and save functions
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pd.DataFrame(pickle.load(f))


# Defining a function to plot smoother plots of validation and accuracy - this often helps to identify a trend better
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points


def model_train(X_train,y_train,X_val,y_val):
    # Importing the Keras libraries and packages
    from keras.utils import Sequence
    import tensorflow as tf
    

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    class DataSequenceGenerator(Sequence):

        def __init__(self, X_train, y_train, batch_size):
            self.X, self.y = X_train, y_train
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.X) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array(batch_X), np.array(batch_y)
    

    dropout_rate = 0.05
    params = {'batch_size': 32}
    training_generator = DataSequenceGenerator(X_train, y_train, **params)
    steps_train = len(X_train)/float(params['batch_size'])
    validation_generator = DataSequenceGenerator(X_val, y_val, **params)
    steps_val = len(X_val)/float(params['batch_size'])

    # define baseline model
    def baseline_model():
        # Initialising the RNN
        regressor = Sequential()

        regressor.add(layers.Conv1D(64, 6, activation='relu',  input_shape=X_train[0].shape)) # Input shape is VERY fiddly. May need to try different things. 
        regressor.add(Dropout(dropout_rate))
        regressor.add(layers.Conv1D(128, 6, activation='relu'))
        regressor.add(layers.GlobalMaxPooling1D())
        regressor.add(Dense(1, activation='sigmoid'))
        print(regressor.summary())

        return regressor

    model = baseline_model()
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','accuracy'])
    no_epochs = 10
    #######################################################################
    config = tf.ConfigProto(device_count={"CPU": 20})
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    #######################################################################
    model.fit_generator(generator=training_generator,steps_per_epoch=steps_train,
                        validation_data=validation_generator,validation_steps=steps_val,
                        epochs=no_epochs,use_multiprocessing=True,workers=-1)

    return model, model.history

dictSpectra = {'O5': 42000, 'O9': 34000, 'B0': 30000, 'B2': 20900, 'B5': 15200,
               'B8': 11400, 'A0': 9790, 'A2': 9000, 'A5': 8180, 'F0': 7300, 
               'F2': 7000, 'F5': 6650, 'F8': 6250, 'G0': 5940, 'G2': 5790, 
               'G5': 5560, 'G8': 5310, 'K0': 5150, 'K2': 4830, 'K5': 4410, 
               'M0': 3840, 'M2': 3520, 'M5': 3170 # 'O5': 42000, 'O5': 42000, 
               }


if __name__ == '__main__':
    # load all spectra in internal memory 
    
    locationPlots = 'CNN_plots_STAR_regression/'

    if not os.path.exists(locationPlots):
        os.makedirs(locationPlots) 
    
    X = load_obj(locationPreprocSpectra+'spectra.pkl')
    y = load_obj(locationPreprocSpectra+'classes.pkl')

    X = np.array(X)
    X = X.reshape((X.shape[0],X.shape[1],1))

    print('Dataset before exclusions: '+str(len(y)))
    
    # exclude stars
    star_indexes = (y[0] == 'STAR  ')
    y = y[star_indexes]
    X = X[star_indexes]
    y = np.vectorize(dictSpectra.get)(np.array(y[1].astype(str).str[0:2]))
    
    idx_matched = (y != None)
    y = y[idx_matched]
    X = X[idx_matched]
    numberTargets = len(list(set(y)))
    
    print('Remaining dataset after exclusions: '+str(len(y)))
    
    #scaling
    sc = MinMaxScaler()
    for i in range(len(X)):
        X[i] = sc.fit_transform(X[i])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    
    model, history = model_train(X_train,y_train,X_val,y_val)

    predictions = model.predict(X_test)
    
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(locationPlots+'plotPredictions')
    plt.close()
    
    # Evaluation 
    plt.close()
    # Now our model is built and trained and tested; we look at results
    # First, the loss and accuracy on training+validation data.
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy (Unsmoothed)')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(locationPlots+'accuracy.png')
    plt.close()
    
    
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(locationPlots+'loss.png')
    plt.close()
    
    plt.figure()
    plt.plot(smooth_curve(history.history['acc']), 'bo', label='Smoothed training acc', alpha=0.5)
    plt.plot(smooth_curve(history.history['val_acc']), 'b', label='Smoothed validation acc')
    plt.title('Training and validation Accuracy (smoothed')
    plt.legend()
    plt.savefig(locationPlots+'accuracy_smoothed.png')
    plt.close()
    
    
    
    plt.figure()
    plt.plot(smooth_curve(history.history['loss']), 'bo', label='Smoothed training loss', alpha=0.5)
    plt.plot(smooth_curve(history.history['val_loss']), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss (Smoothed)')
    plt.legend()
    plt.savefig(locationPlots+'loss_smoothed.png')
    plt.close()
    
    
    # Now let's see the results on Test data; rather than just training and validation sets
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Note that in multi-class and imbalanced classification problems, test accuracy is not an ideal metric')