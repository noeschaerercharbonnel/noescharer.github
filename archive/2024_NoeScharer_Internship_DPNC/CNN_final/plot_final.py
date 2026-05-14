#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:00:05 2024

@author: noeschaerer
"""



import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import backend



import scipy.misc
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





# Import the os module
import os
import time
import random
import pylab
from glob import glob

#doing the roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc



#doing the table of AUC values and colors for plot_ROC_onothers
from tabulate import tabulate
from prettytable import PrettyTable
import pickle 


def parameters_conv():
    
    #TO UNHASHTAG
    os.chdir('/Volumes/Noe')
    #os.chdir('/Users/noeschaerer')
    parameters = open('my_params.txt', mode='r')
    
    types_parameters = ['m1', 'm2', 'dist', 'RA', 'DEC', 'polarization', 'peak_time', 'start_time', 'end_time', 'template_name', 'location']
    stop = True

    parameters_conv = []
    while stop == True:
        #processing the data
        parameters_i = parameters.readline()
        if parameters_i == str(""):
            stop = False
        elif parameters_i[0] == 'm':
            pass
        else :
            parameters_i = parameters_i.split()
            for p in range(len(parameters_i)):
                #changing from str to float
                if p >= 0 and p <= 2:
                    parameters_i[p]= int(parameters_i[p]) 
                elif p >= 3 and p <= 8:
                    parameters_i[p]= float(parameters_i[p])
                else:
                    parameters_i[p]= parameters_i[p]

            #creating a dict for each set of parameters
            parameters_i_dict = {}
            for p in range(len(parameters_i)):
                parameters_i_dict[types_parameters[p]] = parameters_i[p]
            parameters_conv.append(parameters_i_dict)
    
    parameters.close()
    return parameters_conv



def imp_info_param(window_duration=64):
    #extract the parameters
    parameters = parameters_conv()
    
    #record the peak and the associated files
    classification_peaks = []
    classification_files = []
    for i in range(len(parameters)):

        #extracting the i-th paramater data
        parameters_i = parameters[i]
        filename_i = parameters_i['location']
        peaktime_i = parameters_i['peak_time']
        classification_peaks.append(peaktime_i)
        classification_files.append(filename_i)
        
    classification_peaks = np.array(classification_peaks)
    classification_files = np.array(classification_files)
    return classification_peaks, classification_files







def spectrogram_loader(data_names, labels_names, condition='withnoise', comparison='no'):
    #number of files
    n_files = len(data_names)
    events = []
    for i in range(n_files):
        #load file names
        data_name = data_names[i]
        label_name = labels_names[i]
        #load data
        with open(data_name, 'rb') as f_data:
            data_spectrogram_i = np.load(f_data, allow_pickle=True)
        with open(label_name, 'rb') as f_label:
            labels_i = np.load(f_label, allow_pickle=True)

        #cut out frequencies above 1e3 Hz            
        indices = np.arange(0,256,1)
        data_spectrogram_i = np.take(data_spectrogram_i, indices, axis=-1)
        new_shape = data_spectrogram_i.shape[0:3]
        data_spectrogram_i = np.reshape(data_spectrogram_i, new_shape)
        
        #record data
        if i == 0 :
            data = data_spectrogram_i
            labels = labels_i
        elif i > 0 :
            data = np.concatenate((data, data_spectrogram_i))
            labels = np.concatenate((labels, labels_i))
        if i == n_files-1:
            print('----')
        
        for j in range(len(data_spectrogram_i)):
            events.append([data_name, labels[j]])
        
    if comparison == 'no':
        return data, labels
    elif comparison == 'yes':
        return data, labels, events








def data_processing(i_epoch, i_step, data_names, labels_names, index_shuffle_train, index_shuffle_validation, test='yes', init_dir='scale_200', batch_size=16, fraction_train=0.8, window_duration=16, condition='withnoise', plot='no', comparison='no'):
    
    start = time.time()
           
    print('  -----------  ')    
    print('Batch number ' + str(i_step+1))
    
    if comparison == 'no':
        data, labels = spectrogram_loader(data_names, labels_names,  condition='withnoise', comparison=comparison)
    elif comparison == 'yes':
        data, labels, events = spectrogram_loader(data_names, labels_names,  condition='withnoise', comparison=comparison)
    
    #extract the shapes of the data and labels for training and validation
    data_shape = data.shape ; labels_shape = labels.shape
    print('Shapes before any transformation :')
    print(data_shape, labels_shape)
    initial_nfiles = data_shape[0]
    
    length_train = int(data_shape[0]*fraction_train)
            
    
    #shuffling for the first epoch
    if i_epoch == 0:
        index_shuffle = np.arange(0, data_shape[0])
        random.shuffle(index_shuffle)
        index_shuffle_train = index_shuffle[0 : length_train]
        index_shuffle_validation = index_shuffle[length_train : ]
        
    print(len(index_shuffle_train), len(index_shuffle_validation))
    
    data_train = np.array([data[index] for index in index_shuffle_train])
    labels_train = np.array([labels[index] for index in index_shuffle_train], dtype=int)
    data_validation = np.array([data[index] for index in index_shuffle_validation])
    labels_validation = np.array([labels[index] for index in index_shuffle_validation], dtype=int)
    
    #for mdc comparison
    if comparison == 'yes':
        events = [events[index] for index in index_shuffle_validation]
                
    #convert it to readable data for the CNN    
    #random_spec_index = np.random.randint(0, len(data), 8*batch_size)
    #data = np.array([data[i] for i in random_spec_index])
    #labels = np.array([labels[i] for i in random_spec_index])
    #data_shape = data.shape ; labels_shape = labels.shape
    
    data_train = np.expand_dims(data_train, axis=-1) ; data_validation = np.expand_dims(data_validation, axis=-1)
    labels_train = np.expand_dims(labels_train, axis=-1) ; labels_validation = np.expand_dims(labels_validation, axis=-1)
    
    #extract the shapes of the data and labels for training and validation
    data_train_shape = data_train.shape ; labels_train_shape = labels_train.shape
    data_validation_shape = data_validation.shape ; labels_validation_shape = labels_validation.shape
    print('Shapes after shuffling and converting it to readable data for CNNs :')
    print(data_train_shape, labels_train_shape)
    print(data_validation_shape, labels_validation_shape)
    

    #calculate the percentage increase of files thanks do data augmentation
    final_nfiles = data_train_shape[0]
    
    #calculate the percentage increase of files that are a merger
    final_mergers = 0
    for i in range(len(labels_train)):
        label_i = labels_train[i]
        if label_i == 1:
            final_mergers += 1
    
    #calculate the percentage of the files that are mergers
    #percentage_m = (final_mergers - initial_mergers)/final_mergers
    
    
    end = time.time()
    elapsed = end - start
    print('Time spent for data processing and augmentation for batch ' + str(i_step+1) + ' is ' +str(elapsed) +' sec.')
    #print('The percentage of the number of spectrograms containing mergers is > ' + str(int(percentage_m*100)))
    print('  -----------  ')
    
    if comparison == 'no':
        if i_epoch == 0:
            return data_train, labels_train, data_validation, labels_validation, index_shuffle_train, index_shuffle_validation
        elif i_epoch > 0:
            return data_train, labels_train, data_validation, labels_validation
    elif comparison == 'yes':
        if i_epoch == 0:
            return data_train, labels_train, data_validation, labels_validation, index_shuffle_train, index_shuffle_validation, events
        if i_epoch > 0:
            return data_train, labels_train, data_validation, labels_validation, events





def CNN_model(batch_size, dropout_factor, regularization_factor, n_blocks = 1):
    
    n_spec = 8*batch_size
    input_shape=(128, 256, 1)
    
    # Design model
    model = models.Sequential() 
    
    np.random.seed(1)
    tf.random.set_seed(2)
    
    #Architecture
    model.add(layers.Conv2D(16, (8, 8), activation='relu', data_format = 'channels_last', kernel_initializer = 'glorot_uniform', input_shape=input_shape, kernel_regularizer =tf.keras.regularizers.l1(regularization_factor)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Dropout(dropout_factor))
    for i_block in range(2, n_blocks+1):
        model.add(layers.Conv2D(16*i_block, (8, 8), activation='relu', data_format = 'channels_last', kernel_initializer = 'glorot_uniform', kernel_regularizer =tf.keras.regularizers.l1(regularization_factor)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(BatchNormalization(axis = -1))
        model.add(layers.Dropout(dropout_factor))
    #model.add(layers.Conv2D(64,  (16, 25), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(BatchNormalization(axis = -1))
    #model.add(layers.Conv2D(128,  (16, 25), activation='relu'))
    #model.add(BatchNormalization(axis = -1))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu', kernel_initializer = 'glorot_uniform', kernel_regularizer =tf.keras.regularizers.l1(regularization_factor)))
    model.add(layers.Dropout(dropout_factor))
    model.add(layers.Dense(8, activation='relu', kernel_initializer = 'glorot_uniform', kernel_regularizer =tf.keras.regularizers.l1(regularization_factor)))
    model.add(layers.Dropout(dropout_factor))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform'))   
    #metrics
    
    metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
        keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy'), metrics=metrics)  
    
    return model






# IDENTITY BLOCK

def identity_block(X, dropout_factor, reg_factor, f, filters, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = layers.BatchNormalization(axis = -1)(X) # Default axis
    X = Activation('relu')(X)
    X = layers.Dropout(dropout_factor)(X)
    
    ## Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = layers.BatchNormalization(axis = -1)(X) # Default axis
    X = Activation('relu')(X)
    X = layers.Dropout(dropout_factor)(X)

    ## Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = layers.BatchNormalization(axis = -1)(X) # Default axis
    X = layers.Dropout(dropout_factor)(X)
    
    ## Final step: Add shortcut value to main path
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)  

    return X



def convolutional_block(X, dropout_factor, reg_factor, f, filters, s = 2, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = layers.Dropout(dropout_factor)(X)

    ### START CODE HERE
    
    ## Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = layers.Dropout(dropout_factor)(X)

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X) 
    X = BatchNormalization(axis = -1)(X)
    X = layers.Dropout(dropout_factor)(X)
    
    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X_shortcut)
    X_shortcut = BatchNormalization(axis = -1)(X_shortcut)
    X = layers.Dropout(dropout_factor)(X)
    
    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X




def ResNet50(batch_size, dropout_factor, reg_factor, classes = 1, training=True):
    
    n_spec = 8*batch_size
    input_shape = (128, 256, 1)
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (16, 25), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0), use_bias=True, kernel_regularizer =tf.keras.regularizers.l2(reg_factor))(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = layers.Dropout(dropout_factor)(X)
    

    # Stage 2
    X = convolutional_block(X, dropout_factor, reg_factor, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, dropout_factor, reg_factor, 3, [64, 64, 256])
    X = identity_block(X, dropout_factor, reg_factor, 3, [64, 64, 256])
    X = layers.Dropout(dropout_factor)(X)
    
    ## Stage 3
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, dropout_factor, reg_factor, f = 3, filters = [128,128,512], s = 2)
    X = layers.Dropout(dropout_factor)(X)
    
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [128,128,512], initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [128,128,512], initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [128,128,512], initializer=random_uniform)
    X = layers.Dropout(dropout_factor)(X)

    # Stage 4
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024], s = 2)
    X = layers.Dropout(dropout_factor)(X)
    
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024] , initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024] , initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024] , initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024] , initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [256, 256, 1024] , initializer=random_uniform)
    X = layers.Dropout(dropout_factor)(X)

    # Stage 5
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, dropout_factor, reg_factor, f = 3, filters = [512, 512, 2048], s = 2)
    X = layers.Dropout(dropout_factor)(X)
    
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [512, 512, 2048], initializer=random_uniform)
    X = identity_block(X, dropout_factor, reg_factor, f = 3, filters = [512, 512, 2048], initializer=random_uniform)
    X = layers.Dropout(dropout_factor)(X)

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.Dropout(dropout_factor)(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model





def CNN_histories(test='yes', model_choice='simple', n_blocks=2, init_dir='scale_200', window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0):   
    
    #dropout_factor = float(input('Enter the dropout factor : '))
    dropout_factor = 0.0
    #regularization_factor = float(input('Enter the regularization factor : '))
    regularization_factor = 0.0
    
    #choose directory
    os.chdir('/Volumes/Noe')
    if init_dir != 'MDC':
        path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
    elif init_dir == 'MDC':
        init_dir = 'scale_1'
        path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)

        
    path_dir_history = path_dir+'/history'
    files_epochs = glob(path_dir_history+'/history_epoch*')
    n_epoch = len(files_epochs)
    
    file_global = path_dir_history + 'history_global'
    
    
    
    #this model was run for:
    print('This model was run for '+str(n_epoch)+' epochs')
    print('It has the following conditions and structure.')
    print('It has the structure of : '+model_choice)
    print('It was trained on the scale : '+init_dir[-2:])
    print('With or without noise : '+condition)
    print('The window duration is : '+str(window_duration))
    print('It was trained over 1176 files, and '+str(int(len(parameters_conv())*0.7))+' spectrograms')
    print('It has a dropout factor of : '+str(dropout_factor))
    print('It has a regularization factor of : '+str(regularization_factor))
    
    
    #doing over epochs and batches   
    acc_precise = []
    val_acc_precise = []
    loss_precise = []
    val_loss_precise = []
    tpr_precise = []
    fpr_precise = []
    
    #doing over epochs
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    tpr = []
    fpr = []
    
    
    
    n_total = 0
    #n_epoch
    for i_epoch in range(n_epoch):
        
        #load the model at epoch i_epoch
        file_epoch_i = files_epochs[i_epoch]
        #loading the data
        with open(file_epoch_i, 'rb') as f_data:
            history_i = np.load(f_data, allow_pickle=True)
            
        #defining the number of batches
        n_batches = len(history_i)
         
        #n_batches
        for i_batch in range(n_batches):
            
            #define the batch history for the epoch i_epoch
            history_ij = history_i[i_batch]
            acc_ij = history_ij['binary_accuracy'][0]
            val_acc_ij = history_ij['val_binary_accuracy'][0]
            loss_ij = history_ij['loss'][0]
            val_loss_ij = history_ij['val_loss'][0]
            tp = history_ij['true_pos'][0]
            tn = history_ij['true_neg'][0]
            fp = history_ij['false_pos'][0]
            fn = history_ij['false_neg'][0]
            
            #calculating roc curves
            tpr_ij = tp/(tp+fn)
            fpr_ij = fp/(fp+tn)
            
            #record everything
            acc_precise.append(acc_ij)
            val_acc_precise.append(val_acc_ij)
            loss_precise.append(loss_ij)
            val_loss_precise.append(val_loss_ij)
            tpr_precise.append(tpr_ij)
            fpr_precise.append(fpr_ij)
            
            
        #doing a mean over the epoch
        begin_epoch = n_total
        end_epoch = n_total+n_batches
        #meaning
        acc_epoch = acc_precise[begin_epoch:end_epoch]
        val_acc_epoch = val_acc_precise[begin_epoch:end_epoch]
        loss_epoch = loss_precise[begin_epoch:end_epoch]
        val_loss_epoch = val_loss_precise[begin_epoch:end_epoch]
        tpr_epoch = tpr_precise[begin_epoch:end_epoch]
        fpr_epoch = fpr_precise[begin_epoch:end_epoch]
        
        #record the mean over the epoch
        acc.append(np.mean(np.array(acc_epoch)))
        val_acc.append(np.mean(np.array(val_acc_epoch)))
        loss.append(np.mean(np.array(loss_epoch)))
        val_loss.append(np.mean(np.array(val_loss_epoch)))
        tpr.append(np.mean(np.array(tpr_epoch)))
        fpr.append(np.mean(np.array(fpr_epoch)))
        
    
        n_total += n_batches
        
    #make the grids for plots
    grid_precise = np.linspace(1, n_epoch+1, n_total)
    grid = np.arange(1, n_epoch+1, 1)
    
    return acc_precise, val_acc_precise, loss_precise, val_loss_precise, tpr_precise, fpr_precise, grid_precise, acc, val_acc, loss, val_loss, tpr, fpr, grid




def plot_ROC_onitself(n_files=100, test='yes', model_choice='simple', n_blocks=2, window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0):
    
    
    #Recover model structure
    if model_choice=='simple':
        model = CNN_model(n_files, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(n_files, dropout_factor, regularization_factor)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
            keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'), metrics=metrics)
       
    #scales directory
    scales = ['MDC', 'scale_10', 'scale_20', 'scale_50', 'scale_80']
    
    #create figure
    plt.figure(1)

    #create color indexes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_index = 0
    for init_scale in scales:
        #load the model
        os.chdir('/Volumes/Noe')
        if init_scale != 'MDC':
            init_dir = init_scale
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
                #if plot == 'yes':
                    #names_files_ft = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/information_E1_ft_withoutnoise'+init_dir+'*')

            elif condition == 'withnoise':
                names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withnoise_MDC_'+init_dir+'*')
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
        
        
        elif init_scale == 'MDC':
            init_dir = 'scale_1'
            path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/mean_withoutnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/std_withoutnoise_MDC.txt'
               
            elif condition == 'withnoise':
                names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withnoise_MDC_'+init_dir+'*')
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
                
        
        #load model
        dir_models = glob(path_dir+'/Model_cnn*')
        filepath = dir_models[-1]
        model.load_weights(filepath, skip_mismatch=False)
    
         
        #loading the global mean and std
        number_events = len(parameters_conv())  
        with open(file_name_mean, 'r') as f:
            mean = f.readline()
        mean = float(mean[:])
        mean = mean/number_events
         
        #loading the global std
        with open(file_name_std, 'r') as f:
            std = f.readline()
        std = float(std[:])
        std = np.sqrt(std/number_events)
        
        #open the directory shuffling indexes
        path_dir_shuffling = path_dir+'/indexes'
        
        with open(path_dir_shuffling+'/shuffling_names_spectrogram.npy', 'rb') as f_data:
            names_files_spectrogram = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/shuffling_names_labels.npy', 'rb') as f_data:
            names_files_labels = np.load(f_data, allow_pickle=True)   
            
        
        #loading files from the shuffled set        
        with open(path_dir_shuffling+'/indexesshuffling_train.npy', 'rb') as f_data:
                index_shuffle_train_global = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/indexesshuffling_validation.npy', 'rb') as f_data:
                index_shuffle_validation_global = np.load(f_data, allow_pickle=True)
        
        #doing over n_times batches
        n_times = int(n_files/40)
        max_times = int(1176/40)
        
        
        for i in range(n_times):
            
            #loading the data as if i_epoch > 0 
            index_shuffle_train = index_shuffle_train_global[i]
            index_shuffle_validation = index_shuffle_validation_global[i]
            
            #load the files of the batch as if i_epoch > 0
            begin_batch = i*40
            if i == max_times - 1:
                end_batch = -1
            else:
                end_batch = (i+1)*40
            names_files_spectrogram_i, names_files_labels_i = names_files_spectrogram[begin_batch:end_batch], names_files_labels[begin_batch:end_batch]
            data_validation, labels_train, data_i, labels_i = data_processing(1, i, names_files_spectrogram_i, names_files_labels_i, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=n_files, fraction_train=0.7)  
            
            if i == 0:
                data = data_i
                labels = labels_i
            else :
                data = np.concatenate((data, data_i))
                labels = np.concatenate((labels, labels_i))

        #normalizing the data
        data = (data-mean)/std
        
        #calculating the predictions and fpr, tpr, auc
        y_pred_keras = model.predict(data).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
            
        #doing the roc curve
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(labels))]
        ns_auc = roc_auc_score(labels, ns_probs)
        
        # summarize scores
        print('No training: ROC AUC=%.3f' % (ns_auc))
        print('Binary classification: ROC AUC=%.3f' % (auc_keras))
        
        #plot roc curves
        plt.plot(fpr_keras, tpr_keras, label='Trained '+init_scale+' : AUC = {:.3f}'.format(auc_keras), color = colors[color_index])
        color_index +=1
        
    #plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    path_dir_fig = path_dir_pictures+'/ROC'
    if not os.path.exists(path_dir_fig):
        os.mkdir(path_dir_fig)
    
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.title('ROC and AUC')
    plt.savefig(path_dir_fig+'/ROC_curve.png')
    plt.show()
    
 
    
    
    
    
def plot_ROC_onothers(plot='MDC', n_files=100, test='yes', model_choice='simple', n_blocks=2, window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0):
    
    
    #Recover model structure
    if model_choice=='simple':
        model = CNN_model(n_files, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(n_files, dropout_factor, regularization_factor)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
            keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'), metrics=metrics)
       
    #scales directory
    if plot == 'MDC':
        scales_plot = ['MDC_scale_1', 'MDC_scale_2', 'MDC_scale_5', 'MDC_scale_10', 'MDC_scale_20']
    elif plot == 'SCALES':
        scales_plot = ['MDC_scale_1', 'scale_1', 'scale_10', 'scale_20', 'scale_50', 'scale_80']  
        
    #take into account that there is less files for exceptions directories
    exceptions = ['scale_1', 'MDC_scale_2', 'MDC_scale_5', 'MDC_scale_10', 'MDC_scale_20']
    
    #trained models
    scales_model = ['scale_10', 'scale_20', 'scale_50', 'scale_80']
    
    # Create a figure with four subplots
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax11 = ax[0][0]
    ax12 = ax[0][1]
    ax21 = ax[1][0]
    ax22 = ax[1][1]
    
    #create AUC table
    AUC_RECORDS = []

    #create color indexes
    colors = ['red', 'cyan', 'blue', 'green', 'orange', 'purple']
    color_index = 0
    
    #TO DELETE
    scales_plot = ['MDC_scale_2', 'MDC_scale_5']
    scales_model = ['scale_80']
    for init_dir in scales_plot:
        #take into account that there is less files for init_dir in exceptions
        if init_dir in exceptions:
            n_files = int(1176*0.3)+1
        elif init_dir not in exceptions:
            n_files = 1176
            
        #load the files names
        os.chdir('/Volumes/Noe')
        if init_dir[0:3]!= 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
                
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
        
        elif init_dir[0:3]== 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir[4:] +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withoutnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withoutnoise_MDC.txt'
               
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
                
    
         
        #open the directory shuffling indexes
        path_dir_shuffling = path_dir+'/indexes'
        
        with open(path_dir_shuffling+'/shuffling_names_spectrogram.npy', 'rb') as f_data:
            names_files_spectrogram = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/shuffling_names_labels.npy', 'rb') as f_data:
            names_files_labels = np.load(f_data, allow_pickle=True)   
            
        #loading the global mean and std
        number_events = len(parameters_conv())  
        with open(file_name_mean, 'r') as f:
            mean = f.readline()
        mean = float(mean[:])
        mean = mean/number_events
         
        #loading the global std
        with open(file_name_std, 'r') as f:
            std = f.readline()
        std = float(std[:])
        std = np.sqrt(std/number_events)
        
        
        #loading n_files from the shuffled set        
        with open(path_dir_shuffling+'/indexesshuffling_train.npy', 'rb') as f_data:
                index_shuffle_train_global = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/indexesshuffling_validation.npy', 'rb') as f_data:
                index_shuffle_validation_global = np.load(f_data, allow_pickle=True)
        
        #doing over n_times batches
        n_times = int(n_files/40)
        max_times = int(1176/40)
        
        n_times = 2
        for i in range(n_times):
            
            #loading the data as if i_epoch > 0 
            index_shuffle_train = index_shuffle_train_global[i]
            index_shuffle_validation = index_shuffle_validation_global[i]
            
            #load the files of the batch as if i_epoch > 0
            begin_batch = i*40
            if i == max_times - 1:
                end_batch = -1
            else:
                end_batch = (i+1)*40
            names_files_spectrogram_i, names_files_labels_i = names_files_spectrogram[begin_batch:end_batch], names_files_labels[begin_batch:end_batch]
            data_train, labels_train, data_i, labels_i = data_processing(1, i, names_files_spectrogram_i, names_files_labels_i, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=n_files, fraction_train=0.7)  
            
            if i == 0:
                data = data_i
                labels = labels_i
            else :
                data = np.concatenate((data, data_i))
                labels = np.concatenate((labels, labels_i))
                
        #normalizing the data
        data = (data-mean)/std
        
        for scale_test in scales_model:
            
            #load modes
            os.chdir('/Volumes/Noe')
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ scale_test +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            dir_models = glob(path_dir+'/Model_cnn*')
            filepath = dir_models[-1]
            model.load_weights(filepath, skip_mismatch=False)
            
            #calculating the predictions and fpr, tpr, auc
            y_pred_keras = model.predict(data).ravel()
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, y_pred_keras, pos_label=1)
            auc_keras = auc(fpr_keras, tpr_keras)
                
            #doing the roc curve
            # generate a no skill prediction (majority class)
            ns_probs = [0 for _ in range(len(labels))]
            ns_auc = roc_auc_score(labels, ns_probs)
            
            # summarize scores
            print('No training: ROC AUC=%.3f' % (ns_auc))
            print('Binary classification: ROC AUC=%.3f' % (auc_keras))
            
            #plot roc curves
            color_i = colors[color_index]
            if scale_test == 'scale_10':
                ax11.plot(fpr_keras, tpr_keras, color=color_i, label = init_dir)
            elif scale_test == 'scale_20':
                ax12.plot(fpr_keras, tpr_keras, color=color_i)
            elif scale_test == 'scale_50':
                ax21.plot(fpr_keras, tpr_keras, color=color_i)
            elif scale_test == 'scale_80':
                ax22.plot(fpr_keras, tpr_keras, color=color_i)
                
            #record AUC table
            AUC_RECORDS.append({'trained' : scale_test, 'tested' : init_dir, 'auc' : auc_keras, 'color' : color_i})
        
        #change color for new tested datas
        color_index +=1
      
        
    #plot random classifier
    ax11.plot([0, 1], [0, 1], 'k--')
    ax12.plot([0, 1], [0, 1], 'k--')
    ax21.plot([0, 1], [0, 1], 'k--')
    ax22.plot([0, 1], [0, 1], 'k--')
    

    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    path_dir_fig = path_dir_pictures+'/ROC'
    if not os.path.exists(path_dir_fig):
        os.mkdir(path_dir_fig)
        
    #RECORD AUCs
    file_name = path_dir_fig+'/AUC_RECORDS_'+plot+'.npy'
    with open(file_name, 'wb') as f:
        pickle.dump(AUC_RECORDS, f)
    
    #Add labels and titles
    ax11.set_xlabel('False Positive Rate', fontsize = 11)
    ax11.set_ylabel('True Positive Rate', fontsize = 11)
    ax12.set_xlabel('False Positive Rate', fontsize = 11)
    ax12.set_ylabel('True Positive Rate', fontsize = 11)
    ax21.set_xlabel('False Positive Rate', fontsize = 11)
    ax21.set_ylabel('True Positive Rate', fontsize = 11)
    ax22.set_xlabel('False Positive Rate', fontsize = 11)
    ax22.set_ylabel('True Positive Rate', fontsize = 11)
    ax11.title.set_text('ROC and AUC trained on scale 10')
    ax12.title.set_text('ROC and AUC trained on scale 20')
    ax21.title.set_text('ROC and AUC trained on scale 50')
    ax22.title.set_text('ROC and AUC trained on scale 80')
    
    params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
    pylab.rcParams.update(params)

    #changing the space in between plots
    plt.subplots_adjust(left=0.10,
                    bottom=0.10, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    # show the legend
    fig.legend(loc='lower right')
    # show the plot
    fig.suptitle('ROC and AUC trained on different scales for '+plot, fontsize=15)
    fig.savefig(path_dir_fig+'/ROC_curve_global'+plot+'.png')
    plt.show()
    
    return AUC_RECORDS







#PLOT FREQUENCY OF CNN OUTPUT WITH RESPECT TO CNN OUTPUT

def plot_frequency(plot='MDC', n_files=100, test='yes', model_choice='simple', n_blocks=2, window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0):
    
    
    #Recover model structure
    if model_choice=='simple':
        model = CNN_model(n_files, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(n_files, dropout_factor, regularization_factor)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
            keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'), metrics=metrics)
       
    #scales directory
    if plot == 'MDC':
        scales_plot = ['MDC_scale_1', 'MDC_scale_2', 'MDC_scale_5', 'MDC_scale_10', 'MDC_scale_20']
    elif plot == 'SCALES':
        scales_plot = ['MDC_scale_1', 'scale_1', 'scale_10', 'scale_20', 'scale_50', 'scale_80']  
        
    #take into account that there is less files for exceptions directories
    exceptions = ['scale_1', 'MDC_scale_2', 'MDC_scale_5', 'MDC_scale_10', 'MDC_scale_20']
    
    #trained models
    scales_model = ['scale_10', 'scale_20', 'scale_50', 'scale_80']
    
    # Create a figure with four subplots
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax11 = ax[0][0]
    ax12 = ax[0][1]
    ax21 = ax[1][0]
    ax22 = ax[1][1]

    #create color indexes
    colors = ['red', 'cyan', 'blue', 'green', 'orange', 'purple']
    color_index = 0

    for init_dir in scales_plot:
        #take into account that there is less files for init_dir in exceptions
        if init_dir in exceptions:
            n_files = int(1176*0.3)+1
        elif init_dir not in exceptions:
            n_files = 1176
            
        #load the files names
        os.chdir('/Volumes/Noe')
        if init_dir[0:3]!= 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
                
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
        
        elif init_dir[0:3]== 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir[4:] +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withoutnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withoutnoise_MDC.txt'
               
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
                
    
         
        #open the directory shuffling indexes
        path_dir_shuffling = path_dir+'/indexes'
        
        with open(path_dir_shuffling+'/shuffling_names_spectrogram.npy', 'rb') as f_data:
            names_files_spectrogram = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/shuffling_names_labels.npy', 'rb') as f_data:
            names_files_labels = np.load(f_data, allow_pickle=True)   
            
        #loading the global mean and std
        number_events = len(parameters_conv())  
        with open(file_name_mean, 'r') as f:
            mean = f.readline()
        mean = float(mean[:])
        mean = mean/number_events
         
        #loading the global std
        with open(file_name_std, 'r') as f:
            std = f.readline()
        std = float(std[:])
        std = np.sqrt(std/number_events)
        
        
        #loading n_files from the shuffled set        
        with open(path_dir_shuffling+'/indexesshuffling_train.npy', 'rb') as f_data:
                index_shuffle_train_global = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/indexesshuffling_validation.npy', 'rb') as f_data:
                index_shuffle_validation_global = np.load(f_data, allow_pickle=True)
        
        #doing over n_times batches
        n_times = int(n_files/40)
        max_times = int(1176/40)
        
        for i in range(n_times):
            
            #loading the data as if i_epoch > 0 
            index_shuffle_train = index_shuffle_train_global[i]
            index_shuffle_validation = index_shuffle_validation_global[i]
            
            #load the files of the batch as if i_epoch > 0
            begin_batch = i*40
            if i == max_times - 1:
                end_batch = -1
            else:
                end_batch = (i+1)*40
            names_files_spectrogram_i, names_files_labels_i = names_files_spectrogram[begin_batch:end_batch], names_files_labels[begin_batch:end_batch]
            data_train, labels_train, data_i, labels_i = data_processing(1, i, names_files_spectrogram_i, names_files_labels_i, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=n_files, fraction_train=0.7)  
            
            if i == 0:
                data = data_i
                labels = labels_i
            else :
                data = np.concatenate((data, data_i))
                labels = np.concatenate((labels, labels_i))
                
        #normalizing the data
        data = (data-mean)/std
        
        for scale_test in scales_model:
            
            #load modes
            os.chdir('/Volumes/Noe')
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ scale_test +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            dir_models = glob(path_dir+'/Model_cnn*')
            filepath = dir_models[-1]
            model.load_weights(filepath, skip_mismatch=False)
            
            #calculating the predictions and fpr, tpr, auc
            y_pred_keras = model.predict(data).ravel()
            
            #plot roc curves
            color_i = colors[color_index]
            if scale_test == 'scale_10':
                ax11.hist(y_pred_keras, density=True, bins=30, histtype='step', range=(0, 1), color=color_i, label = init_dir)
            elif scale_test == 'scale_20':
                ax12.hist(y_pred_keras, density=True, bins=30, histtype='step', range=(0, 1), color=color_i)
            elif scale_test == 'scale_50':
                ax21.hist(y_pred_keras, density=True, bins=30, histtype='step', range=(0, 1), color=color_i)
            elif scale_test == 'scale_80':
                ax22.hist(y_pred_keras, density=True, bins=30, histtype='step', range=(0, 1), color=color_i)
                
        #change color for new tested datas
        color_index +=1
    

    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    path_dir_fig = path_dir_pictures+'/ROC'
    if not os.path.exists(path_dir_fig):
        os.mkdir(path_dir_fig)
        
    
    #Add labels and titles
    ax11.set_xlabel('CNN output', fontsize = 11)
    ax11.set_ylabel('Frequency of the output', fontsize = 11)
    ax12.set_xlabel('CNN output', fontsize = 11)
    ax12.set_ylabel('Frequency of the output', fontsize = 11)
    ax21.set_xlabel('CNN output', fontsize = 11)
    ax21.set_ylabel('Frequency of the output', fontsize = 11)
    ax22.set_xlabel('CNN output', fontsize = 11)
    ax22.set_ylabel('Frequency of the output', fontsize = 11)
    ax11.title.set_text('Frequencies trained on scale 10')
    ax12.title.set_text('Frequencies trained on scale 20')
    ax21.title.set_text('Frequencies trained on scale 50')
    ax22.title.set_text('Frequencies trained on scale 80')
    
    params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
    pylab.rcParams.update(params)

    #changing the space in between plots
    plt.subplots_adjust(left=0.10,
                    bottom=0.10, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    # show the legend
    fig.legend(loc='lower right')
    # show the plot
    fig.suptitle('Frequency of the CNN output for '+plot, fontsize=15)
    fig.savefig(path_dir_fig+'/Frequency'+plot+'.png')
    plt.show()
    




   

    
#PLOT THE TABLE OF VALUES OF AUC AND COLORS/LABELS
def plot_table(plot='MDC'):
    
    path_file = '/Volumes/Noe/pictures/ROC/AUC_RECORDS_'+plot+'.npy'
    #loading AUC RECORD FILE       
    with open(path_file, 'rb') as f:
        AUC_RECORDS = pickle.load(f)
    
    #{'trained' : scale_test, 'tested' : init_dir, 'auc' : auc_keras, 'color' : color_i}
    trained_scale10 = [] ; trained_scale20 = [] ; trained_scale50 = [] ; trained_scale80 = []
    for i in range(len(AUC_RECORDS)):
        record_i = AUC_RECORDS[i]
        record_i_input = [record_i['tested'], record_i['auc']]
        if record_i['trained'] == 'scale_10':
            trained_scale10.append(record_i_input)
        if record_i['trained'] == 'scale_20':
            trained_scale20.append(record_i_input)
        if record_i['trained'] == 'scale_50':
            trained_scale50.append(record_i_input)
        if record_i['trained'] == 'scale_80':
            trained_scale80.append(record_i_input)
     
    # Specify the Column Names while initializing the Table 
    myTable = PrettyTable(["Tested scales", "Trained scale 10", "Trained scale 20", "Trained scale 50", "Trained scale 80"])
    myTable.title = 'AUC for '+plot
    if plot =='MDC':
        myTable.add_row(["MDC scale 1", str(round(trained_scale10[0][1], 2)), str(round(trained_scale20[0][1], 2)), str(round(trained_scale50[0][1], 2)), str(round(trained_scale80[0][1], 2))])
        myTable.add_row(["MDC scale 2", str(round(trained_scale10[1][1], 2)), str(round(trained_scale20[1][1], 2)), str(round(trained_scale50[1][1], 2)), str(round(trained_scale80[1][1], 2))])
        myTable.add_row(["MDC scale 5", str(round(trained_scale10[2][1], 2)), str(round(trained_scale20[2][1], 2)), str(round(trained_scale50[2][1], 2)), str(round(trained_scale80[2][1], 2))])
        myTable.add_row(["MDC scale 10", str(round(trained_scale10[3][1], 2)), str(round(trained_scale20[3][1], 2)), str(round(trained_scale50[3][1], 2)), str(round(trained_scale80[3][1], 2))])
        myTable.add_row(["MDC scale 20", str(round(trained_scale10[4][1], 2)), str(round(trained_scale20[4][1], 2)), str(round(trained_scale50[4][1], 2)), str(round(trained_scale80[4][1], 2))])
    elif plot == 'SCALES':
        myTable.add_row(["MDC scale 1", str(round(trained_scale10[0][1], 2)), str(round(trained_scale20[0][1], 2)), str(round(trained_scale50[0][1], 2)), str(round(trained_scale80[0][1], 2))])
        myTable.add_row(["scale 1", str(round(trained_scale10[1][1], 2)), str(round(trained_scale20[1][1], 2)), str(round(trained_scale50[1][1], 2)), str(round(trained_scale80[1][1], 2))])
        myTable.add_row(["scale 10", str(round(trained_scale10[2][1], 2)), str(round(trained_scale20[2][1], 2)), str(round(trained_scale50[2][1], 2)), str(round(trained_scale80[2][1], 2))])
        myTable.add_row(["scale 20", str(round(trained_scale10[3][1], 2)), str(round(trained_scale20[3][1], 2)), str(round(trained_scale50[3][1], 2)), str(round(trained_scale80[3][1], 2))])
        myTable.add_row(["scale 50", str(round(trained_scale10[4][1], 2)), str(round(trained_scale20[4][1], 2)), str(round(trained_scale50[4][1], 2)), str(round(trained_scale80[4][1], 2))])
        myTable.add_row(["scale 80", str(round(trained_scale10[5][1], 2)), str(round(trained_scale20[5][1], 2)), str(round(trained_scale50[5][1], 2)), str(round(trained_scale80[5][1], 2))])
    print(myTable)
    
 
    
 
    
 
    
 
    
 
    
def comparison_snr_ml(plot='MDC', n_files=100, test='yes', model_choice='simple', n_blocks=2, window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0): 

        
    #Recover model structure
    if model_choice=='simple':
        model = CNN_model(n_files, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(n_files, dropout_factor, regularization_factor)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
            keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'), metrics=metrics)
       
    #scales directory
    scales_plot = ['MDC_scale_1', 'MDC_scale_2', 'MDC_scale_5', 'MDC_scale_10', 'MDC_scale_20']
            
    #trained models
    scales_model = ['scale_10', 'scale_20', 'scale_50']
    
    # Create a figure with four subplots
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax11 = ax[0][0]
    ax12 = ax[0][1]
    ax21 = ax[1][0]
    ax22 = ax[1][1]

    #create color indexes
    colors = ['red', 'cyan', 'green', 'orange', 'purple']
    color_index = 0
    
    for init_dir in scales_plot:

        n_files = int(1176*0.3)+1
            
        #keep track of events
        events_global = []

        #take into account that there is less files for init_dir in exceptions
            
        #load the files names
        os.chdir('/Volumes/Noe')
        if init_dir[0:3]!= 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
                
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
        
        elif init_dir[0:3]== 'MDC':
            path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir[4:] +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            if condition == 'withoutnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withoutnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withoutnoise_MDC.txt'
               
            elif condition == 'withnoise':
                file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
                file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir[4:] +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
                
    
        names_files_spectrogram = np.array(glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram*'))
        names_files_labels = np.array(glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/labels*'))
        names_files_spectrogram = names_files_spectrogram[0:n_files] ;  names_files_labels = names_files_labels[0:n_files]
            
        #loading the global mean and std
        number_events = len(parameters_conv())  
        with open(file_name_mean, 'r') as f:
            mean = f.readline()
        mean = float(mean[:])
        mean = mean/number_events
         
        #loading the global std
        with open(file_name_std, 'r') as f:
            std = f.readline()
        std = float(std[:])
        std = np.sqrt(std/number_events)
        

        
        #doing over n_times batches
        n_times = int(n_files/40)
        max_times = int(1176/40)
        
        for i in range(n_times):
            
            #loading the data as if i_epoch = 0
            index_shuffle_train = 0
            index_shuffle_validation = 0
            
            #load the files of the batch as if i_epoch > 0
            begin_batch = i*40
            if i == max_times - 1:
                end_batch = -1
            else:
                end_batch = (i+1)*40
            names_files_spectrogram_i, names_files_labels_i = names_files_spectrogram[begin_batch:end_batch], names_files_labels[begin_batch:end_batch]
            
            #load the data
            data_train, labels_train, data_i, labels_i, index_shuffle_train, index_shuffle_validation, events = data_processing(0, i, names_files_spectrogram_i, names_files_labels_i, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=n_files, fraction_train=0.0, comparison='yes')  
            
            #recording the data
            if i == 0:
                data = data_i
                labels = labels_i
                    
            else :
                data = np.concatenate((data, data_i))
                labels = np.concatenate((labels, labels_i))
               
            events_global.extend(events)


                
        #normalizing the data
        data = (data-mean)/std
                
        events_global_files = []
        for j in range(len(events_global)):
            events_global_files.append(events_global[j][0])
            
        
        for scale_test in scales_model:
            
            #load modes
            os.chdir('/Volumes/Noe')
            path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ scale_test +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
            dir_models = glob(path_dir+'/Model_cnn*')
            filepath = dir_models[-1]
            model.load_weights(filepath, skip_mismatch=False)
            
            #calculating the predictions and fpr, tpr, auc
            y_pred_keras = model.predict(data).ravel()
            
            #finding detections
            arg_detections = np.argwhere(y_pred_keras > 0.5).flatten()
            true_detections = []
            for arg in arg_detections:
                true_event_label = events_global[arg][1]
                if true_event_label == 1:
                    true_detections.append(events_global[arg][0])
            
            
            #count the number of detections per file
            final_count_list = []
            final_files_list = []
            check = []
            for j in range(len(true_detections)):
                detection_file_j = true_detections[j]
                if detection_file_j not in check:
                    count_j = true_detections.count(detection_file_j)
                    count_files_j = events_global_files.count(detection_file_j)/2
                    fraction = count_j/count_files_j
                    final_count_list.append(fraction)
                    final_files_list.append(detection_file_j)
                check.append(detection_file_j)
                

            #go back to number of the file and not name of it
            number_files = []
            for j in range(len(final_count_list)):
                file_j = final_files_list[j]
                number_file_j = np.argwhere(names_files_spectrogram == file_j)[0][0]
                number_files.append(number_file_j)
                
            #add the files where there is no events detected
            names_files_spectrogram = names_files_spectrogram[0:n_times*40]
            for j in range(len(names_files_spectrogram)):
                name_file_spec_j = names_files_spectrogram[j]
                if name_file_spec_j not in final_files_list:
                    final_count_list.append(0)
                    number_files.append(j)
                
            #sort them by ascending values of the number's file
            new_number_files = sorted(number_files)
            new_final_count_list = []
            for j in new_number_files:
                position_j = np.argwhere(np.array(number_files) == j)[0][0]
                new_final_count_list.append(final_count_list[position_j])
            
            #meaning out
            final_count_list = []
            for j in range(int(len(new_number_files)/50)+1):
                begin = j*50 
                if j < int(len(number_files)/50):
                    end = (j+1)*50
                elif j == int(len(number_files)/50):
                    end = -1
                fraction_batch_j = new_final_count_list[begin : end]
                fraction_mean_j = np.mean(np.array(fraction_batch_j))
                final_count_list.append(fraction_mean_j)
                
            #creating the final arrays 
            number_files_final = np.arange(1, int(len(new_number_files)/50)*50, 50)
            number_files_final = np.concatenate((number_files_final, np.array([new_number_files[-1]])))
            
            #plot curves
            color_i = colors[color_index]
            if scale_test == 'scale_10':
                ax12.plot(number_files_final, final_count_list, label=init_dir, color=color_i)
            elif scale_test == 'scale_20':
                ax21.plot(number_files_final, final_count_list, color=color_i)
            elif scale_test == 'scale_50':
                ax22.plot(number_files_final, final_count_list, color=color_i)
                
            
                
        #change color for new tested datas
        color_index +=1
    
    
    
    #DO THE SAME FOR MF
    path_file = '/Volumes/Noe/pictures/global/rapport/Sarah/mdc_snr.npy'
    #loading snr mdc FILE       
    with open(path_file, 'rb') as f:
        mdc_snr = np.load(f)
        
    path_file = '/Volumes/Noe/pictures/global/rapport/Sarah/mf_snr_detections.npy'
    #loading snr mdc FILE       
    with open(path_file, 'rb') as f:
        our_snr = np.load(f)

    

    #do the same for snr matched filtering
    snr_count = []
    number_files = []
    check = []
    snr_count_i = 0
    n_files_i = 0
    for i in range(len(mdc_snr)):
        #extract and convert the data
        snr_i = mdc_snr[i]
        snr_value_i = float(snr_i[-1])
        number_file_i = float(snr_i[0][7:])
        #count the snr
        if i>0 and number_file_i not in check:
            #record the data
            if snr_count_i == 0:
                fraction = 0
            elif snr_count_i > 0:
                fraction = snr_count_i/n_files_i
            snr_count.append(fraction)
            number_files.append(number_file_i)
            snr_count_i = 0
            n_files_i = 0
        #put a threshold on snr value
        if snr_value_i > 10:
            snr_count_i += 1
        #record to keep track of which file
        check.append(number_file_i)
        n_files_i +=1
        
    #do the same for snr matched filtering, with already a threshold of 10
    our_number_files = []
    for i in range(len(our_snr)):
        #extract and convert the data
        snr_i = our_snr[i]
        number_file_i = float(snr_i[0][29:-4])
        our_number_files.append(number_file_i)
    
    our_snr_count = []
    for i in range(len(number_files)):
        number_file_i = number_files[i]
        count_events_i = number_files.count(number_file_i)
        count_total_i = check.count(number_file_i)
        fraction = count_events_i/count_total_i
        our_snr_count.append(fraction)
     
    #average out for 50 files
    snr_mean = []
    our_snr_mean = []
    for i in range(int(len(number_files)/50)+1):
        begin = i*50 
        if i < int(len(number_files)/50):
            end = (i+1)*50
        elif i == int(len(number_files)/50):
            end = -1
        snr_batch_i = snr_count[begin : end]
        our_snr_batch_i = our_snr_count[begin : end]
        snr_mean_i = np.mean(np.array(snr_batch_i))
        our_snr_mean_i = np.mean(np.array(our_snr_batch_i))
        snr_mean.append(snr_mean_i) ; our_snr_mean.append(our_snr_mean_i)
        
    #creating the final arrays 
    number_files_final = np.arange(1, int(len(number_files)/50)*50, 50)
    number_files_final = np.concatenate((number_files_final, np.array([number_files[-1]])))
    
    #plot the averaged values lines
    ax11.plot(number_files_final, snr_mean, label='optimal SNR : '+str(int(np.mean(snr_mean)*100))+'%', color='black')
    ax11.plot(number_files_final, our_snr_mean, label='computed SNR : '+str(int(np.mean(our_snr_mean)*100))+'%' , color='blue')
    
    
    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    path_dir_fig = path_dir_pictures+'/ROC'
    if not os.path.exists(path_dir_fig):
        os.mkdir(path_dir_fig)
        
    
    #Add labels and titles
    ax11.set_ylim(-0.05, 1.05)
    ax12.set_ylim(-0.05, 1.05)
    ax21.set_ylim(-0.05, 1.05)
    ax22.set_ylim(-0.05, 1.05)
    ax11.set_xlabel('Number of the file', fontsize = 11)
    ax11.set_ylabel('Fraction of mergers found', fontsize = 11)
    ax12.set_xlabel('Number of the file', fontsize = 11)
    ax12.set_ylabel('Fraction of mergers found', fontsize = 11)
    ax21.set_xlabel('Number of the file', fontsize = 11)
    ax21.set_ylabel('Fraction of mergers found', fontsize = 11)
    ax22.set_xlabel('Number of the file', fontsize = 11)
    ax22.set_ylabel('Fraction of mergers found', fontsize = 11)
    ax11.title.set_text('Fractions with matched filtering')
    ax12.title.set_text('Fractions trained on scale 10')
    ax21.title.set_text('Fractions trained on scale 20')
    ax22.title.set_text('Fractions trained on scale 50')
    
    params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 12),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
    pylab.rcParams.update(params)

    #changing the space in between plots
    plt.subplots_adjust(left=0.10,
                    bottom=0.10, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    # show the legend
    fig.legend(loc='lower right')
    # show the plot
    fig.suptitle('Fractions of the mergers found for MF and ML', fontsize=15)
    fig.savefig(path_dir_fig+'/Fractions.png')
    plt.show()
        
        
    
        
    
def matched_filtering(): 
    path_file = '/Volumes/Noe/pictures/global/rapport/Sarah/mdc_snr.npy'
    #loading snr mdc FILE       
    with open(path_file, 'rb') as f:
        mdc_snr = np.load(f)
        
    path_file = '/Volumes/Noe/pictures/global/rapport/Sarah/mf_snr_detections.npy'
    #loading snr mdc FILE       
    with open(path_file, 'rb') as f:
        our_snr = np.load(f)
        

    bns = 0 ; bhns = 0 ; bbh = 0
    n_mergers = len(our_snr) 
    for i in range(n_mergers):
        type_merger = int(our_snr[i][-2])
        if type_merger == 1:
            bns += 1
        elif type_merger == 2:
            bhns += 1
        elif type_merger == 3:
            bbh += 1
    
    bns = int((bns/n_mergers)*100) ; bhns = int((bhns/n_mergers)*100) ; bbh = int((bbh/n_mergers)*100)
        
    print('BNS = '+str(bns) +'%') ; print('BHNS = '+str(bhns) +'%') ; print('BBH = '+str(bbh) +'%')
    

    #create figure
    plt.figure(1)
    
    #do the same for snr matched filtering
    snr_count = []
    number_files = []
    check = []
    snr_count_i = 0
    n_files_i = 0
    for i in range(len(mdc_snr)):
        #extract and convert the data
        snr_i = mdc_snr[i]
        snr_value_i = float(snr_i[-1])
        number_file_i = float(snr_i[0][7:])
        #count the snr
        if i>0 and number_file_i not in check:
            #record the data
            if snr_count_i == 0:
                fraction = 0
            elif snr_count_i > 0:
                fraction = snr_count_i/n_files_i
            snr_count.append(fraction)
            number_files.append(number_file_i)
            snr_count_i = 0
            n_files_i = 0
        #put a threshold on snr value
        if snr_value_i > 10:
            snr_count_i += 1
        #record to keep track of which file
        check.append(number_file_i)
        n_files_i +=1
        
    #do the same for snr matched filtering, with already a threshold of 10
    our_number_files = []
    for i in range(len(our_snr)):
        #extract and convert the data
        snr_i = our_snr[i]
        number_file_i = float(snr_i[0][29:-4])
        our_number_files.append(number_file_i)
    
    our_snr_count = []
    for i in range(len(number_files)):
        number_file_i = number_files[i]
        count_events_i = number_files.count(number_file_i)
        count_total_i = check.count(number_file_i)
        fraction = count_events_i/count_total_i
        our_snr_count.append(fraction)
        

     
    #average out for 50 files
    snr_mean = []
    our_snr_mean = []
    for i in range(int(len(number_files)/50)+1):
        begin = i*50 
        if i < int(len(number_files)/50):
            end = (i+1)*50
        elif i == int(len(number_files)/50):
            end = -1
        snr_batch_i = snr_count[begin : end]
        our_snr_batch_i = our_snr_count[begin : end]
        snr_mean_i = np.mean(np.array(snr_batch_i))
        our_snr_mean_i = np.mean(np.array(our_snr_batch_i))
        snr_mean.append(snr_mean_i) ; our_snr_mean.append(our_snr_mean_i)
        
    #creating the final arrays 
    number_files_final = np.arange(1, int(len(number_files)/50)*50, 50)
    number_files_final = np.concatenate((number_files_final, np.array([number_files[-1]])))
        
    #plot the averaged values lines
    plt.plot(number_files_final, snr_mean, label='optimal SNR : '+str(int(np.mean(snr_mean)*100))+'%', color='black')
    plt.plot(number_files_final, our_snr_mean, label='computed SNR : '+str(int(np.mean(our_snr_mean)*100))+'%' , color='blue')
    #include 0 in the plot
    plt.ylim(-0.05, 1.05)
    
    
    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    path_dir_fig = path_dir_pictures+'/ROC'
    if not os.path.exists(path_dir_fig):
        os.mkdir(path_dir_fig)
        
    
    # show the plot
    plt.xlabel('Number of the file')
    plt.ylabel('Fraction of mergers found')
    plt.legend()
    plt.title('Fractions of the mergers found with MF for SNR values > 10')
    plt.savefig(path_dir_fig+'/Fractions_MF.png')
    plt.show()
        
    
        
        
        
        
        
from keras.utils import plot_model       
        
        
def plot_model(plot='MDC', n_files=100, test='yes', model_choice='simple', n_blocks=2, window_duration=16, condition='withnoise', dropout_factor=0.0, regularization_factor=0.0): 

        
    #Recover model structure
    if model_choice=='simple':
        model = CNN_model(n_files, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(n_files, dropout_factor, regularization_factor)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'), keras.metrics.TruePositives(name='true_pos'), keras.metrics.TrueNegatives(name='true_neg'),
            keras.metrics.FalseNegatives(name='false_neg'), keras.metrics.FalsePositives(name='false_pos'),]
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'), metrics=metrics)
        
    #change dir
    os.chdir('/Volumes/Noe')
    path_dir_pictures = '/Volumes/Noe/pictures'
    resume = model.summary()
    
    
        
        
        
    
    
    
    
#plot = input('Enter if you want to do MDC or SCALES plot : ')
plot='MDC'
#dropout_factor = float(input('Enter the dropout factor : '))
dropout_factor = 0.0
#regularization_factor = float(input('Enter the regularization factor : '))
regularization_factor = 0.0
#input3 = input('Test = yes / No test = no : ')
input3 = 'yes'
#input4 = input('Model choice = simple/Resnet50 : ')
input4 = 'Resnet50'
#input5 = int(input('n_blocks : '))
input5 = 4
#input7 = input('Write scale_number : ')
input7 = 'scale_80'
#input8 = input('Condition = withnoise/withoutnoise : ')
input8 = 'withnoise'
#input9 = int(input('Size of the window : '))
input9 = 16

#plot ROC CURVES WITH AUC
#n_files = int(input('Write the number of files over which you want to do the ROC curve (>100)'))
n_files = 1176


#plot_ROC_onitself(n_files=n_files, test=input3, model_choice=input4, n_blocks=input5, window_duration=input9, condition=input8, dropout_factor=dropout_factor, regularization_factor=regularization_factor)


os.chdir('/Volumes/Noe')
path_dir_pictures = '/Volumes/Noe/pictures'
if not os.path.exists(path_dir_pictures):
    os.mkdir(path_dir_pictures)
    
#AUC_RECORDS = plot_ROC_onothers(plot=plot, n_files=n_files, test=input3, model_choice=input4, n_blocks=input5, window_duration=input9, condition=input8, dropout_factor=dropout_factor, regularization_factor=regularization_factor)
    
#plot_frequency(plot=plot, n_files=n_files, test=input3, model_choice=input4, n_blocks=input5, window_duration=input9, condition=input8, dropout_factor=dropout_factor, regularization_factor=regularization_factor)

#comparison_snr_ml(plot=plot, n_files=n_files, test=input3, model_choice=input4, n_blocks=input5, window_duration=input9, condition=input8, dropout_factor=dropout_factor, regularization_factor=regularization_factor)

matched_filtering()

#plot='SCALES'
#plot_table(plot=plot)

#plot_model(plot=plot, n_files=n_files, test=input3, model_choice=input4, n_blocks=input5, window_duration=input9, condition=input8, dropout_factor=dropout_factor, regularization_factor=regularization_factor)



scales = ['MDC', 'scale_10', 'scale_20']       
thresholds = ['80', '88', '90', '92', '95']


# Create a figure with four subplots
fig, ax =plt.subplots(1, 2, figsize = (12, 6))
ax1 = ax[0]
ax2 = ax[1]

#create a directory
path_dir_fig = path_dir_pictures+'/ROC'
if not os.path.exists(path_dir_fig):
    os.mkdir(path_dir_fig)
        
#create color indexes
colors = ['red', 'cyan', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'black', 'grey']

#len(scales)
for i in range(0):
    
    #load data for the i-th scale
    input7 = scales[i]
    threshold = thresholds[i]
    acc_precise, val_acc_precise, loss_precise, val_loss_precise, tpr_precise, fpr_precise, grid_precise, acc, val_acc, loss, val_loss, tpr, fpr, grid = CNN_histories(test=input3, model_choice = input4, n_blocks=input5, init_dir=input7, condition=input8, window_duration=input9, dropout_factor=dropout_factor, regularization_factor=regularization_factor)

    ax1.plot(grid, acc, label='acc '+input7, color=colors[2*i])
    ax1.plot(grid, val_acc, label='val. acc '+input7, color=colors[2*i+1])
    ax1.set_xlabel('Epoch', fontsize = 11)
    ax1.set_ylabel('Accuracy', fontsize = 11)
    ax1.title.set_text('Accuracy with epoch precision')


    ax2.plot(grid, loss, label='loss '+input7, color=colors[2*i+1])
    ax2.plot(grid, val_loss, label='val. loss '+input7, color=colors[2*i])
    ax2.set_xlabel('Epoch', fontsize = 11)
    ax2.set_ylabel('Loss', fontsize = 11)
    ax2.set_ylim(0,2)
    ax2.title.set_text('Loss with epoch precision')
    


fig.legend(loc='lower right')
fig.suptitle('Evolution of metrics during training for each scale', fontsize=15)
fig.savefig(path_dir_fig+'/metrics.png')

    





