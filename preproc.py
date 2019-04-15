
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import gc
import glob
import os

# load and save functions
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pd.DataFrame(pickle.load(f))



def preproc_data(locationSpectra,n_nodes=100):
    
    """
    df_current['information'].iloc[0] = class_
    df_current['information'].iloc[1] = subclass_
    df_current['information'].iloc[2] = z_
    df_current['information'].iloc[3] = z_err
    df_current['information'].iloc[4] = z_warn
    df_current['information'].iloc[5] = best_obj
    df_current['information'].iloc[6] = instrument
    
    """

    print('Starting reading the names of the files containing spectra.....')
    filenames = glob.glob(locationSpectra+'*pkl')
    print('Finished.......................................................')
    print('Total length of dataset read is '+str(len(filenames))+' spectra.')
    
    scale_length = 1
    X = np.zeros((int(len(filenames)/scale_length),n_nodes))
    y = [[] for i in range(int(len(filenames)/scale_length))]
    #X = [[[] for i in range(n_nodes)] for j in range(int(len(filenames)/scale_length))]
    # from sklearn.preprocessing import MinMaxScaler
    # sc = MinMaxScaler()

    # guess the range of the wavelengths
    min_X = 3500
    max_X = 11000
    _,step = np.linspace(min_X,max_X, num=n_nodes,retstep=True)
    
    number_files_toRead = int(len(filenames)/scale_length)
    
    for i in range(number_files_toRead):
        
        # print checkpoints
        if(i % int(number_files_toRead/100) == 0):
            print('Progress is '+str(round(100*i/number_files_toRead))+' %')
        
        # read data
        df_current = load_obj(filenames[i])
        l = len(df_current['model'])
        wavelength = np.power(10,df_current['loglam'][0:l])
        flux = np.array(df_current['model'][0:l])
        # flux_scaled = np.array(sc.fit_transform(np.array(flux).reshape(-1,1))).reshape(flux.shape)

        # process data
        tempFlux = [[] for i in range(n_nodes)]

        for j in range(len(flux)):
            if(wavelength[j]>0):
                tempFlux[int((wavelength[j] - min_X) / step)].append(flux[j])


        for j in range(n_nodes):
            tempFlux[j] = np.array(tempFlux[j])
            if(tempFlux[j].size > 0):
                X[i][j] = np.mean(np.array(tempFlux[j]))
            else:
                X[i][j] = 0.

        # now store the classification info
        y.extend((df_current['information'].iloc[0], df_current['information'].iloc[1], df_current['information'].iloc[2], 
            df_current['information'].iloc[3], df_current['information'].iloc[4], df_current['information'].iloc[5],
            df_current['information'].iloc[6]))

        gc.collect()

    return X,y


if __name__ == '__main__':

    locationSpectra = '../spectraClassification/spectra_matched_multiproc/'
    locationData = 'preprocessedData/'

    X,y = preproc_data(locationSpectra,n_nodes=200)

    # save all to pkl file
    if not os.path.exists(locationData):
        os.makedirs(locationData)

    save_obj(X,locationData+'spectra')
    save_obj(y,locationData+'classes')
