#Packages for CNN

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import backend


import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
#from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


#Packages for spectrograms

from scipy import signal
from scipy.fft import fftshift
from pycbc import fft
import matplotlib.pyplot as plt



#Packages for GWF

from pycbc import frame
from pycbc import types
from pycbc import fft


from pycbc.waveform import get_td_waveform, fd_approximants, td_approximants
import pylab


from glob import glob








# Import the os module
import os
import time

# Print the current working directory
print("Previous working directory: {0}".format(os.getcwd()))

# Change the current working directory
#os.chdir('/Volumes/Noe/mysignal_E1')
os.chdir('/Volumes/Noe/scale_80')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

onlyfiles = next(os.walk(os.getcwd()))[2] #directory is your directory path as string
number_files = len(onlyfiles)
print('number of files in ' + str(os.getcwd()) + ' = ' + str(number_files))





#PRODUCTION OF SPECTROGRAM

from gwpy import spectrogram
from gwpy.timeseries import TimeSeries






def parameters_conv():
    
    os.chdir('/Volumes/Noe')
    files_names = glob('parameters_files/*')
    types_parameters = ['#','starting_time', 'time@Maximum', 'redshifted_mass1', 'redshifted_mass2', 'spin1', 'spin1x', 'spin1y', 'spin1z', 'spin2', 'spin2x', 'spin2y', 'spin2z', 'lambda1', 'lambda2', 'z', 'dist', 'ra', 'declination', 'polarization', 'inclination', 'initial phase',  'snr_optimal', 'type_merger']
    
    numbers = []
    for i in range(len(files_names)):
        #read parameters file
        filename = files_names[i]
        parameters = open(filename, mode='r')
        stop = True
        parameters_conv_file = []
        while stop == True:
            #processing the data
            parameters_j = parameters.readline()
            if parameters_j == str(""):
                stop = False
            else :
                parameters_j = parameters_j.split()
                for p in range(len(parameters_j)):
                    #changing from str to float
                    if p == 0 or p == (len(parameters_j)-1):
                        parameters_j[p]= int(parameters_j[p]) 
                    else :
                        parameters_j[p]= float(parameters_j[p])
        
                #creating a dict for each set of parameters
                parameters_j_dict = {}
                for p in range(len(parameters_j)):
                    parameters_j_dict[types_parameters[p]] = parameters_j[p]
            #testing if the element was already recorded
            number_i = parameters_j_dict['#']
            if number_i not in numbers:
                numbers.append(number_i)
                parameters_conv_file.append(parameters_j_dict)
                
        #recording for that file and globaly
        parameters_conv_file = np.array(parameters_conv_file)
        if i == 0:
            parameters_global = parameters_conv_file
        else:
            parameters_global = np.concatenate((parameters_global, parameters_conv_file)) 
        
    return parameters_global

print(len(parameters_conv()))



def imp_info_param(window_duration=64):
    #extract the parameters
    parameters = parameters_conv()
    
    #record the peak and the associated files
    classification_peaks = []
    classification_files = []
    for i in range(len(parameters)):

        #extracting the i-th paramater data
        parameters_i = parameters[i]
        peaktime_i = parameters_i['time@Maximum']
        #finding the file
        number_file = int((peaktime_i - 1000000000)/2048)
        start_i = 1000000000+2048*number_file
        filename_i = 'E-E1_STRAIN_DATA-'+str(start_i)+'-2048.gwf'
        #recording peaktime and associated file
        classification_peaks.append(peaktime_i)
        classification_files.append(filename_i)
        
    classification_peaks = np.array(classification_peaks)
    classification_files = np.array(classification_files)
    return classification_peaks, classification_files







def create_spectrogram_complex(data_i, peaks_i, start_i, window_duration=64):
        
    #finding how much you need to data augment 
    n_step = int(2048/window_duration)
    
    #creating the original labels array
    labels_i_test = []
    l_event_test = 0
    for j in range(n_step):
        #defining start and retrieving information
        start_time_j = j * window_duration
        end_time_j = (j+1) * window_duration
        peak_time_l = peaks_i[l_event_test]
                
        if start_time_j <= peak_time_l <= end_time_j :
            labels_i_test.append(1)
            l_event_test += 1
        
        else :
            labels_i_test.append(0)
            
        if l_event_test == len(peaks_i):
            l_event_test = 0

    labels_i_test = np.array(labels_i_test)
    
    #finding important info for data augmentation
    n_nonmergers = len(np.where(labels_i_test == 0)[0])
    n_oldmergers = len(np.where(labels_i_test == 1)[0])
    n_newspec_permerger_i = int(n_nonmergers/n_oldmergers)
    
    #recording info
    data_spectrogram_i = [] ; ft = []
    labels_i = []
    
    l_event = 0
    for j in range(n_step):
        #defining start and retrieving information
        start_time_j = j * window_duration
        end_time_j = (j+1) * window_duration
        peak_time_l = peaks_i[l_event]
                
        if start_time_j <= peak_time_l <= end_time_j :
            
            for k in range(0, n_newspec_permerger_i+1):
                #beginning of the array
                if j == 0 :
                    #spectrogram
                    start_global_k = 0
                    start_k = start_global_k + int(8192 * window_duration * k/n_newspec_permerger_i)
                    data_ij = data_i[start_k : start_k + 8192*window_duration]
                    
                #end of the array
                elif j == int(n_step)-1:
                    
                    #spectrogram
                    start_global_k = - int(8192 * window_duration)
                    start_k = start_global_k + int(8192 * window_duration * (k/n_newspec_permerger_i))
                    data_ij = data_i[8192 * window_duration * j + start_k: 8192 * window_duration * (j+1) + start_k]
                    
                #elsewhere in the array
                else: 
                    #spectrogram
                    #print(k)
                    start_global_k = int(8192*(peak_time_l - window_duration))
                    start_k = start_global_k + int(8192 * window_duration * k/n_newspec_permerger_i) 
                    data_ij = data_i[start_k : start_k + 8192*window_duration]
                    
                data_ij = (data_ij - data_ij.mean())/data_ij.std()
                spec = data_ij.spectrogram2(fftlength=0.25, overlap = 0.125)**(1/2.)
                f = spec.frequencies ; t = spec.times
                #f = np.array(f) ; t=np.array(t)
                data_spectrogram_i.append(np.absolute(spec))
                ft.append([f,t])
                labels_i.append(1)
            
            l_event += 1
        
        else :
            
            #no data augmentation
            data_ij = data_i[8192 * window_duration * j: 8192 * window_duration * (j+1)]
            data_ij = (data_ij - data_ij.mean())/data_ij.std()
            spec = data_ij.spectrogram2(fftlength=0.25, overlap = 0.125)**(1/2.)
            f = spec.frequencies ; t = spec.times
            #f = np.array(f) ; t=np.array(t)
            data_spectrogram_i.append(np.absolute(spec))
            ft.append([f,t])
            labels_i.append(0)
        
        if l_event == len(peaks_i):
            l_event = 0
    
    print(np.array(data_spectrogram_i).shape, np.array(labels_i).shape)
    print('percentage of mergers = ' + str(int(len(np.where(np.array(labels_i)== 1)[0])*100/len(labels_i))))
    return data_spectrogram_i, ft, labels_i






def create_spectrogram_simple(data_i, noise_i, peaks_i, start_i, window_duration=64):
        
    #finding how much you need to data augment 
    n_step = int(2048/window_duration)
    
    #recording info
    data_spectrogram_i = [] ; ft = []
    labels_i = []
    
    l_event = 0
    
    for j in range(n_step):
        #defining start and retrieving information
        start_time_j =  j * window_duration
        end_time_j =  (j+1) * window_duration
        peak_time_l = peaks_i[l_event] - start_i
                
        if start_time_j <= peak_time_l <= end_time_j :
            
            #defining the window around the peak merger
            if j == 0 or j == n_step-1:
                begin_index_data = j*window_duration*8192
                end_index_data = (j+1)*window_duration*8192
                begin_index_noise = begin_index_data
                end_index_noise = end_index_data
                
            elif j>=1:
                begin_index_data = 8192 * (int(peak_time_l-window_duration/2))
                end_index_data = 8192 * (int(peak_time_l+window_duration/2))
                
                if j == 1:
                    begin_index_noise = end_index_data
                    end_index_noise = end_index_data + 8192*window_duration
                else:
                    begin_index_noise = begin_index_data - 8192*window_duration
                    end_index_noise = begin_index_data
                
                
            #no data augmentation : spectrogram of data
            labels_i.append(1)
            data_ij = data_i[begin_index_data : end_index_data]
            spec = data_ij.spectrogram2(fftlength=0.25, overlap = 0.125)**(1/2.)
            spec = np.absolute(spec)
            f = spec.frequencies ; t = spec.times
            data_spectrogram_i.append(spec)
            ft.append([f,t])
            mean_i_data = np.array(spec.mean())
            std_i_data = np.sum(np.array((spec - spec.mean())**2)) 
            
            #taking a spectrogram of noise
            labels_i.append(0)
            noise_ij = noise_i[begin_index_noise : end_index_noise]
            spec = noise_ij.spectrogram2(fftlength=0.25, overlap = 0.125)**(1/2.)
            spec = np.absolute(spec)
            f = spec.frequencies ; t = spec.times
            data_spectrogram_i.append(spec)
            ft.append([f,t])
            mean_i_noise = np.array(spec.mean())
            std_i_noise = np.sum(np.array((spec - spec.mean())**2))
            
            #recording both mean and st
            if l_event == 0:
                mean_i = mean_i_data + mean_i_noise
                std_i = std_i_data + std_i_noise
            else:
                mean_i = mean_i + mean_i_data + mean_i_noise
                std_i = std_i + std_i_data + std_i_noise
            
            l_event += 1
        
        if l_event == len(peaks_i):
            break
    
    print(np.array(data_spectrogram_i).shape, np.array(labels_i).shape)
    print('percentage of mergers = ' + str(int(len(np.where(np.array(labels_i)== 1)[0])*100/len(labels_i))))
    return data_spectrogram_i, ft, labels_i, mean_i, std_i








def production_file(init_dir, data_i, noise_i, peaks_i, start_i, test='yes', condition='withnoise', window_duration=64):
    
    if test == 'no':
        data, ft, labels_i = create_spectrogram_complex(data_i, peaks_i, start_i, window_duration=window_duration)
    elif test == 'yes':
        data, ft, labels_i, mean_i, std_i = create_spectrogram_simple(data_i, noise_i, peaks_i, start_i, window_duration=window_duration)
        
    if condition == 'withnoise':
        
        #change directory
        os.chdir('/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test)

        #create a file for the spectrogram of the i-th data file
        file_name = 'spectrogram_E1-withnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, data, allow_pickle=True, fix_imports=True)
            file.close()
            
        #create a file for the spectrogram of the i-th data file with the frequency and time information
        file_name = 'information_E1_ft_withnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, ft, allow_pickle=True, fix_imports=True)
            file.close()
            
        file_name = 'labels_withnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, labels_i, allow_pickle=True, fix_imports=True)
            file.close()
        
        #saving the mean
        file_name = 'mean_withnoise_MDC.txt'
        if not os.path.exists(file_name):
            mean = mean_i
            with open(file_name, 'w') as f:
                f.write(str(mean))
                f.close()
        else:
            with open(file_name, 'r') as f:
                previous_mean = f.readline()
            mean = float(previous_mean[:]) + mean_i
            with open(file_name, 'w') as f:
                f.write(str(mean))
                f.close()
            
        #saving the mean
        file_name = 'std_withnoise_MDC.txt'
        if not os.path.exists(file_name):
            std = std_i
            with open(file_name, 'w') as f:
                f.write(str(std))
                f.close()
        else:
            with open(file_name, 'r') as f:
                previous_std = f.readline()
            std = float(previous_std[:]) + std_i
            with open(file_name, 'w') as f:
                f.write(str(std))
                f.close()
                
            
    elif condition == 'withoutnoise':
        
        #change directory
        os.chdir('/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+init_dir +'-'+str(window_duration)+test)
        
        #create a file for the spectrogram of the i-th noise file
        
        file_name = 'spectrogram_E1-withoutnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, data, allow_pickle=True, fix_imports=True)
            file.close()
            
        #create a file for the spectrogram of the i-th data file with the frequency and time information
        file_name = 'information_E1_ft_withoutnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, ft, allow_pickle=True, fix_imports=True)
            file.close()
            
        file_name = 'labels_withoutnoise_MDC_'+ init_dir + str(start_i) + '-2048-' + str(window_duration) +'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, labels_i, allow_pickle=True, fix_imports=True)
            file.close()
        
        #saving the mean
        file_name = 'mean_withoutnoise_MDC.txt'
        if not os.path.exists(file_name):
            mean = mean_i
            with open(file_name, 'w') as f:
                f.write(str(mean))
                f.close()
        else:
            with open(file_name, 'r') as f:
                previous_mean = f.readline()
            mean = float(previous_mean[:]) + mean_i
            with open(file_name, 'w') as f:
                f.write(str(mean))
                f.close()
            
        #saving the mean
        file_name = 'std_withoutoise_MDC.txt'
        if not os.path.exists(file_name):
            std = std_i
            with open(file_name, 'w') as f:
                f.write(str(std))
                f.close()
        else:
            with open(file_name, 'r') as f:
                previous_std = f.readline()
            std = float(previous_std[:]) + std_i
            with open(file_name, 'w') as f:
                f.write(str(std))
                f.close()
    






def production_directory(test='yes', n_files_produce=100, init_dir = 'scale_80', condition = 'withnoise', window_duration=64):
    start_time= time.time()
    
    if condition == 'withnoise':
        #create a directory
        os.chdir('/Volumes/Noe')
        path_dir = 'DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        onlyfiles = next(os.walk(path_dir))[2]
        
    elif condition == 'withoutnoise':
        #create a directory
        os.chdir('/Volumes/Noe')
        path_dir = 'DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        onlyfiles = next(os.walk(path_dir))[2]
    
    number_files_spec = len(onlyfiles)
    print('There is '+str(int(number_files_spec/3))+' files in the directory')
    print(' ----------- ')
    
    #Reading the files
    start = 1000000000
    
    #retrieving the labels to do data augmentation
    peaks, OGfiles = imp_info_param(window_duration=window_duration)
    
    noise_files = glob('/Volumes/Noe/E1_NOISE/E1/*')
    
    if number_files_spec == 0:
        start_file = 0
    elif number_files_spec > 0:
        start_file = int((number_files_spec-2)/3)
    for i in range(start_file, n_files_produce):
        
        print('Production fichier numéro : '+str(i+1))
        start_time_i = time.time()
        #extracting the i-th file of the folder
        start_i = start + i*2048
        
        #go to the MY_signal_E1 directory
        file_name_i = '/Volumes/Noe/E1/'+'E-E1_STRAIN_DATA-'+str(start_i)+'-2048.gwf'

        data_i = TimeSeries.read(file_name_i, 'E1:STRAIN')
        
        noise_i = 0
        
        if test == 'yes' or condition == 'withoutnoise':
            
            #go to the noise directory and take the right file
            #considering I only have a certain fraction of the noise files
            if i < len(noise_files):
                j = i
            
            if  len(noise_files) <= i <= 2*len(noise_files)-1 :
                j = i -len(noise_files)
                
            elif i >= 2*len(noise_files) :
                
                j = i-2*len(noise_files)
             
            file_name_j = noise_files[j]
            noise_i = TimeSeries.read(file_name_j, 'E1:STRAIN')
            
        if condition == 'withoutnoise':
            #substract noise from data
            data_i = np.array(data_i)
            noise_i = np.array(noise_i)
            data_i -= noise_i
            data_i = TimeSeries(data_i, t0 = start_i, sample_rate=8192)
            
        if init_dir != 'scale_1':
            if len(init_dir) == 7:
                scale_factor = int(init_dir[-1])
            elif len(init_dir) > 7:
                scale_factor = int(init_dir[-2:])
            data_i = (np.array(data_i) - np.array(noise_i))*scale_factor + np.array(noise_i)
            data_i = TimeSeries(data_i, t0 = start_i, sample_rate = 8192)
                
        #extract corresponding peaks
        index_peaks_i = np.where('E-E1_STRAIN_DATA-'+str(start_i)+'-2048.gwf' == OGfiles)[0]
        peaks_i = peaks[index_peaks_i]
        peaks_i = np.sort(peaks_i)
        
        #create files in the directory
        production_file(init_dir, data_i, noise_i, peaks_i, start_i, test=test, condition=condition, window_duration=window_duration)
        
        #calculatin the elapsed time for the i-th signal
        end_time_i = time.time()
        elapsed_i = end_time_i - start_time_i
        print('Temps d\'exécution par fichier en sec =', elapsed_i)
        print(' ----------- ')
    
    print(' ----------- ')
    print('Nombre de fichiers produits = '+str(i+1))
        
    #calculatin the elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    print('Temps d\'exécution total en min =', elapsed/60)





#input1 = input('Test = yes / No test = no : ')
input1 = 'yes'
#input2 = int(input('Number of files to produce : '))
input2 = int(1176*0.3)+1
#input3 = input('Write scale_number : ')
input3 = 'scale_2'
#input4 = input('Condition, withnoise or withoutnoise : ')
input4 = 'withnoise'
#input5 = int(input('Size of the window : '))
input5 = 16


#production_directory(test=input1, n_files_produce=input2, init_dir = input3, condition=input4, window_duration=input5)





