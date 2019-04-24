
import numpy as np
import pandas as pd
import pickle
import gc
import glob
import os
import multiprocessing
import time

#locationSpectra = '../spectraClassification/spectra_matched_multiproc/'
locationSpectra = 'spectra/'
locationData = 'preprocessedDataFull/'
locationDataOneByOne = 'preprocessedDataFullNew/'
n_nodes=500

# load and save functions
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pd.DataFrame(pickle.load(f))



def preproc_data(varyingData, constantData ):
    
    """
    df_current['information'].iloc[0] = class_
    df_current['information'].iloc[1] = subclass_
    df_current['information'].iloc[2] = z_
    df_current['information'].iloc[3] = z_err
    df_current['information'].iloc[4] = z_warn
    df_current['information'].iloc[5] = best_obj
    df_current['information'].iloc[6] = instrument
    """
    
    [locationSpectrum, counter] = varyingData
    [numberFilenames, locationDataOneByOne, step] = constantData
    if(counter%int(numberFilenames/10) == 0):
        print('progress is: ' +str(round(100*counter/numberFilenames,1))+ ' %')
    
    X = np.zeros(n_nodes) 
    # read data
    try:
        df_current = load_obj(locationSpectrum)
        
        
        l = len(df_current['model'])
        wavelength = np.power(10,df_current['loglam'][0:l])
        flux = np.array(df_current['model'][0:l])
        # flux_scaled = np.array(sc.fit_transform(np.array(flux).reshape(-1,1))).reshape(flux.shape)
    
        # process data
        tempFlux = [[] for j in range(n_nodes)]
    
        for j in range(len(flux)):
            if(wavelength[j]>0):
                tempFlux[int((wavelength[j] - min_X) / step)].append(flux[j])
    
    
        for j in range(n_nodes):
            tempFlux[j] = np.array(tempFlux[j])
            if(tempFlux[j].size > 0):
                X[j] = np.mean(np.array(tempFlux[j]))
            else:
                X[j] = 0.
    
        # now store the classification info
        y = np.array([df_current['information'].iloc[0], df_current['information'].iloc[1], df_current['information'].iloc[2], 
            df_current['information'].iloc[3], df_current['information'].iloc[4], df_current['information'].iloc[5],
            df_current['information'].iloc[6]])
        
        resultArray = np.array([X,y])
        save_obj(resultArray,locationDataOneByOne+str(counter))
        gc.collect()
        return resultArray
    except:
        # occasionally it cannot open some files
        return 0



if __name__ == '__main__':

    print('Starting reading the names of the files containing spectra.....')
    filenames = glob.glob(locationSpectra+'*pkl')
    print('Finished.......................................................')
    print('Total length of dataset read is '+str(len(filenames))+' spectra.')



    freeProc = 1
    n_proc=multiprocessing.cpu_count()-freeProc

    counter = list(range(len(filenames)))
    varyingData = []
    for i in range(len(filenames)):
        varyingData.append([filenames[i], counter[i]])
        
    # guess the range of the wavelengths
    min_X = 3500
    max_X = 11000
    _,step = np.linspace(min_X,max_X, num=n_nodes,retstep=True)
    
    constantData = [len(filenames),locationDataOneByOne,step]
    if not os.path.exists(locationDataOneByOne):
        os.makedirs(locationDataOneByOne)
    
    import itertools
    initialTime = time.time()
    with multiprocessing.Pool(processes=n_proc) as pool:
        result_list=pool.starmap(preproc_data, zip(varyingData,itertools.repeat(constantData)))
        pool.close()
    elapsedTime = time.time() - initialTime
    print('The processing speed was '+str(round(elapsedTime/len(filenames),2)) +
          ' seconds per file using ' + str(n_proc) + ' processors.')
    print('Total time elapsed was '+ str(round(elapsedTime,2))+ ' seconds')
    
    X = np.zeros((len(result_list),n_nodes))
    linesXToDelete = []
    y = []
    for i in range(len(result_list)):
        try:
            y.append(result_list[i][1])
            X[i] = result_list[i][0]
        except:
            linesXToDelete.append(i)
            pass
    
    X = np.delete(X,linesXToDelete,axis=0)
    # save all to pkl file
    if not os.path.exists(locationData):
        os.makedirs(locationData)

    save_obj(X,locationData+'spectra')
    save_obj(y,locationData+'classes')
