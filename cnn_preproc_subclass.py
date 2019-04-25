#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:08:56 2019

@author: vladgriguta
"""

occurancePercentage = 0.05

locationPreprocSpectra = 'preprocessedData/'
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



def encode_data(y):
    # encode class values as integers
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y,encoder

def decode(dummy_y,encoder):
    """
    Function that takes the dummy variable and its encoder and transforms it
    back to the initial form
    """
    # from dummy back to class names
    encoded_y = np.zeros(len(dummy_y))
    for i in range(len(dummy_y)):
        encoded_y[i] = int(np.argmax(dummy_y[i]))
    classes_y =  encoder.inverse_transform(encoded_y.astype(int))
    
    return classes_y,encoded_y


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


def plot_confusion_matrix(cm, target_names, location):
    
    # neither of these work
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    plt.rcParams["axes.axisbelow"] = True
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap='Blues')
    # plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    """
    #ax.set_xticks(target_names)
    ax.set_xticklabels( [''] + target_names )
    #ax.set_yticks(target_names)
    ax.set_yticklabels( [''] + target_names )
    """
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=target_names, yticklabels=target_names,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(location+'ConfusionMatrix')
    plt.close()
    return ax

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
        model = Sequential()

        model.add(layers.Conv1D(64, 6, activation='relu',  input_shape=X_train[0].shape)) # Input shape is VERY fiddly. May need to try different things. 
        model.add(Dropout(dropout_rate))
        model.add(layers.Conv1D(128, 6, activation='sigmoid'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(Dense(numberTargets, activation='softmax'))
        print(model.summary())

        return model

    model = baseline_model()
    model.compile(optimizer=RMSprop(lr=0.0100), loss='categorical_crossentropy',metrics=['acc'])
    no_epochs = 10
    #######################################################################
    config = tf.ConfigProto(device_count={"CPU": 20})
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    #######################################################################
    model.fit_generator(generator=training_generator,steps_per_epoch=steps_train,
                        validation_data=validation_generator,validation_steps=steps_val,
                        epochs=no_epochs,use_multiprocessing=True,workers=-1)

    return model, model.history



if __name__ == '__main__':
    # load all spectra in internal memory 
    

    locationPlots = 'CNN_plots_galaxy&qso/'

    if not os.path.exists(locationPlots):
        os.makedirs(locationPlots) 
    
    X = load_obj(locationPreprocSpectra+'spectra.pkl')
    y = load_obj(locationPreprocSpectra+'classes.pkl')

    X = np.array(X)
    X = X.reshape((X.shape[0],X.shape[1],1))

    print('Dataset before exclusions: '+str(len(y)))
    
    # exclude stars
    gal_indexes = (y[0] != 'STAR  ')
    y = y[gal_indexes]
    X = X[gal_indexes]
    y = np.array(y[0] +' '+ y[1])
    
    # exclude sparse data
    unique, counts = np.unique(y, return_counts=True)
    dict_counts = dict(zip(unique, counts))
    
    # compute the occurance limit by adjusting the expected number of samples
    # in each class with the occurance percentage
    occuranceLimit = int(occurancePercentage * len(y) / len(unique))
    sparse_indexes = []
    for i in range(len(y)):
        # if the class appears less often than 50 times
        if(dict_counts[y[i]] < occuranceLimit):
            sparse_indexes.append(i)
            
    # eliminate current element from y and X
    y = np.delete(y,sparse_indexes,axis=0)
    X = np.delete(X,sparse_indexes,axis=0)
    
    print('Remaining dataset after exclusions: '+str(len(y)))        
    
    dummy_y,encoder_y = encode_data(y)
    

    # scale x
    sc = MinMaxScaler()
    for i in range(len(X)):
        X[i] = sc.fit_transform(X[i])    

    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2,  random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    
    # Enter target names
    target_names = list(set(y))
    numberTargets = len(target_names)

    
    model, history = model_train(X_train,y_train,X_val,y_val)

    predictions = model.predict_proba(X_test)
    predicted_labels = predictions.argmax(axis=1) # Converts probabilities (e.g. 0.035 0.001 0.704 0.260) to labels (e.g. 0 0 1 0)
    
    
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
    cm = confusion_matrix(y_test.argmax(axis=1), predicted_labels)
    print("Confusion matrix:\n{}".format(cm))

    y_true = y_test.argmax(axis=1)
    #cm_analysis(y_true, predicted_labels, filename=location_plots+'confMatrix.png', labels=[0,1,2])
    plot_confusion_matrix(cm, target_names, location=locationPlots)
    plt.close()
