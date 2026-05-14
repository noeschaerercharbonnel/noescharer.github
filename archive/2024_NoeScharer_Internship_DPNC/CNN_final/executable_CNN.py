#Packages for CNN

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
import pylab

from glob import glob




# Import the os module
import os
import time




import random




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








def spectrogram_loader(data_names, labels_names, condition='withnoise'):
    #number of files
    n_files = len(data_names)
    for i in range(n_files):
        #load file names
        data_name = data_names[i]
        label_name = labels_names[i]
        #load data
        with open(data_name, 'rb') as f_data:
            data_spectrogram_i = np.load(f_data, allow_pickle=True)
        with open(label_name, 'rb') as f_label:
            labels_i = np.load(f_label, allow_pickle=True)
            
        indices = np.arange(0,256,1)
        data_spectrogram_i = np.take(data_spectrogram_i, indices, axis=-1)
        new_shape = data_spectrogram_i.shape[0:3]
        data_spectrogram_i = np.reshape(data_spectrogram_i, new_shape)
        
        #recreate 2d array for labels
        #labels_final = []
       # for j in range(len(labels_i)):
           # label_ij = labels_i[j]
           # if label_ij == 0:
                #labels_final.append([1, 0])
            #elif label_ij == 1:
               # labels_final.append([0, 1])
        #labels_final = np.array(labels_final)
        
        #record data
        if i == 0 :
            data = data_spectrogram_i
            labels = labels_i
        elif i > 0 :
            data = np.concatenate((data, data_spectrogram_i))
            labels = np.concatenate((labels, labels_i))
        if i == n_files-1:
            print('----')
    return data, labels








def data_processing(i_epoch, i_step, data_names, labels_names, index_shuffle_train, index_shuffle_validation, test='yes', init_dir='scale_200', batch_size=16, fraction_train=0.8, window_duration=16, condition='withnoise', plot='no'):
    
    start = time.time()
           
    print('  -----------  ')    
    print('Batch number ' + str(i_step+1))
    
    data, labels = spectrogram_loader(data_names, labels_names,  condition='withnoise')
    
    
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
        index_shuffle_validation = index_shuffle[length_train:]
    
    data_train = np.array([data[index] for index in index_shuffle_train])
    labels_train = np.array([labels[index] for index in index_shuffle_train], dtype=int)
    data_validation = np.array([data[index] for index in index_shuffle_validation])
    labels_validation = np.array([labels[index] for index in index_shuffle_validation], dtype=int)
        

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
    #percentage_mergers = final_mergers/final_nfiles
    
    #calculate the percentage of the files that are mergers
    #percentage_m = (final_mergers - initial_mergers)/final_mergers
    
    
    end = time.time()
    elapsed = end - start
    print('Time spent for data processing and augmentation for batch ' + str(i_step+1) + ' is ' +str(elapsed) +' sec.')
    #print('The percentage of the number of spectrograms containing mergers, thanks to data augmentation is ' + str(int(percentage_mergers*100)))
    #print('The percentage of the number of spectrograms containing mergers is > ' + str(int(percentage_m*100)))
    print('  -----------  ')
    
    
    if i_epoch == 0:
        return data_train, labels_train, data_validation, labels_validation, index_shuffle_train, index_shuffle_validation
    if i_epoch > 0:
        return data_train, labels_train, data_validation, labels_validation








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









def CNN_compiler(step_0, n_epoch, MDC_ornot = 'no', threshold=0.95, test='yes', model_choice='simple', n_blocks=2, n_files=220, init_dir='scale_200', window_duration=64, condition='withnoise', batch_size=10, fraction_train=0.8, input_shape=(128, 1025, 1)):   
    
    #dropout_factor = float(input('Enter the dropout factor : '))
    dropout_factor = 0.0
    #regularization_factor = float(input('Enter the regularization factor : '))
    regularization_factor = 0.0
    
    #create a directory for models saving
    os.chdir('/Volumes/Noe')
    if MDC_ornot == 'no':
        path_dir = 'DIRECTORY_models/'+model_choice+'/'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
        
        if condition == 'withoutnoise':
            names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withoutnoise'+init_dir+'*')
            names_files_labels = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/labels_withoutnoise_'+ init_dir + '*')
            file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
            file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
            #if plot == 'yes':
                #names_files_ft = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/information_E1_ft_withoutnoise'+init_dir+'*')

        elif condition == 'withnoise':
            names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withnoise'+init_dir+'*')
            names_files_labels = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/labels_withnoise_'+ init_dir + '*')
            file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise.txt'
            file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise.txt'
            #if plot == 'yes':
                #names_files_ft = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/information_E1_ft_withnoise'+init_dir+'*')
    elif MDC_ornot == 'yes':
        path_dir = 'DIRECTORY_models/'+model_choice+'/MDC'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)
        if condition == 'withoutnoise':
            names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withoutnoise_MDC_'+init_dir+'*')
            names_files_labels = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/labels_withoutnoise_MDC_'+ init_dir + '*')
            file_name_mean = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
            file_name_std = '/Volumes/Noe/DIRECTORY_withoutnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
            #if plot == 'yes':
                #names_files_ft = glob('/Volumes/Noe/DIRECTORY_withoutnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/information_E1_ft_withoutnoise'+init_dir+'*')
        
        elif condition == 'withnoise':
            names_files_spectrogram = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/spectrogram_E1-withnoise_MDC_'+init_dir+'*')
            names_files_labels = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/labels_withnoise_MDC_'+ init_dir + '*')
            file_name_mean = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/mean_withnoise_MDC.txt'
            file_name_std = '/Volumes/Noe/DIRECTORY_withnoise_E1_MDC_'+ init_dir +'-'+str(window_duration)+test+'/std_withnoise_MDC.txt'
            #if plot == 'yes':
                #names_files_ft = glob('/Volumes/Noe/DIRECTORY_withnoise_E1_'+ init_dir +'-'+str(window_duration)+test+'/information_E1_ft_withnoise'+init_dir+'*')

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    dir_files_epochs = glob(path_dir+'/Model_cnn*')
    epoch_0 = len(dir_files_epochs)
        
    #create a directory for history
    path_dir_history = path_dir+'/history'
    if not os.path.exists(path_dir_history):
        os.mkdir(path_dir_history)
        
    #create a directory for shuffling indexes
    path_dir_shuffling = path_dir+'/indexes'
    if not os.path.exists(path_dir_shuffling):
        os.mkdir(path_dir_shuffling)
    
    
    # Design model
    if model_choice=='simple':
        model = CNN_model(batch_size, dropout_factor, regularization_factor, n_blocks=n_blocks)

    elif model_choice=='Resnet50':
        model = ResNet50(batch_size, dropout_factor, regularization_factor)
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
    
    print(model.summary())
      
    #shuffle the files
    #if starting from epoch 0, creating the shuffling of the files
    names_files_spectrogram = names_files_spectrogram[0:n_files] ; names_files_labels = names_files_labels[0:n_files]
    if epoch_0 == 0:
        #loading the files
        temp = list(zip(names_files_spectrogram, names_files_labels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        #unziping
        names_files_spectrogram, names_files_labels = list(res1), list(res2) 
    
        file_name = path_dir_shuffling+'/shuffling_names_spectrogram.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, np.array(names_files_spectrogram), allow_pickle=True, fix_imports=True)
            file.close()
        
        file_name = path_dir_shuffling+'/shuffling_names_labels.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, np.array(names_files_labels), allow_pickle=True, fix_imports=True)
            file.close()
    
    #otherwise, loading the shuffling of the files to recreate the same batches
    elif epoch_0 > 0:
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
    
        
    
    
    #recursive calls of the 2d CNN
    #number of recursive calls
    n_step = int(n_files/batch_size)
    print('There is '+ str(n_step) + ' batches of size ' + str(batch_size) + ' for n_files = '+str(n_files))
    print(' --------------- ')
    
    #creating lists of indexes for the shuffling
    index_shuffle_train_global = []
    index_shuffle_validation_global = []
    #open shuffling indexes of epoch_0 >= 1, which is saved for steps > 1
    if epoch_0 >= 1 :
        with open(path_dir_shuffling+'/indexesshuffling_train.npy', 'rb') as f_data:
            index_shuffle_train_global = np.load(f_data, allow_pickle=True)
        with open(path_dir_shuffling+'/indexesshuffling_validation.npy', 'rb') as f_data:
            index_shuffle_validation_global = np.load(f_data, allow_pickle=True)
    
    history_global = []
    #doing over epochs
    #n_epoch
    max_step = int(1176/batch_size)
    
    for i_epoch in range(epoch_0, n_epoch):
        
        #setting the recording of history
        history_i = []
        
        #setting the threshhold for early stopping
        count = 0
        end_running = False
                
        #doing over batches
        for i_step in range(step_0, n_step):
            
            #extract the corresponding files
            begin_batch = i_step*batch_size
            if i_step == max_step-1:
                end_batch = -1
                
            else:
                end_batch = (i_step+1)*batch_size
            data_names = names_files_spectrogram[begin_batch:end_batch]
            labels_names = names_files_labels[begin_batch:end_batch]
            
            #shuffle for the first epoch = 0
            if i_epoch == 0:
                
                #including a random unuseful index_shuffle
                index_shuffle_train = 0 ; index_shuffle_validation = 0
                data_train, labels_train, data_validation, labels_validation, index_shuffle_train, index_shuffle_validation = data_processing(i_epoch, i_step, data_names, labels_names, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=batch_size, fraction_train=fraction_train)            
                index_shuffle_train_global.append(index_shuffle_train) ; index_shuffle_validation_global.append(index_shuffle_validation)
                
            #extracting the shuffling for the i_epoch > 0
            elif i_epoch > 0:
                
                #extracting the shuffles for the batch i_step
                index_shuffle_train = index_shuffle_train_global[i_step] ; index_shuffle_validation = index_shuffle_validation_global[i_step]
                data_train, labels_train, data_validation, labels_validation = data_processing(i_epoch, i_step, data_names, labels_names, index_shuffle_train, index_shuffle_validation, test=test, init_dir=init_dir, condition=condition, window_duration=window_duration, batch_size=batch_size, fraction_train=fraction_train)            
            
            #normalize the data
            data_train = (data_train - mean)/std ; data_validation = (data_validation - mean)/std
            
            
            if i_epoch == 0:
                #call the weights of the CNN
                if i_step >= 1:
                    os.chdir('/Volumes/Noe')
                    filepath = path_dir+'/Model_cnn'+model_choice+'_'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'epoch0'+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)+'.keras'
                    model.load_weights(filepath, skip_mismatch=False)
            else:
                os.chdir('/Volumes/Noe')
                if i_step == 0:
                    filepath = path_dir+'/Model_cnn'+model_choice+'_'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'epoch'+str(i_epoch-1)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)+'.keras'
                    model.load_weights(filepath, skip_mismatch=False)
                if i_step > 0:
                    filepath = path_dir+'/Model_cnn'+model_choice+'_'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'epoch'+str(i_epoch)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)+'.keras'
                    model.load_weights(filepath, skip_mismatch=False)
                    
            #y_train = model.predict(data_train, batch_size=10, verbose='auto', steps=None, callbacks=None)
            #y_val = model.predict(data_validation, batch_size=10, verbose='auto', steps=None, callbacks=None)
            #print(y_train)
            print(' --------- ')
            #print(y_val)
            print(' --------- ')


            #fit model
            history_model = model.fit(
                            data_train,
                            labels_train,
                            batch_size=10,
                            epochs=1,
                            # We pass some validation for
                            # monitoring validation loss and metrics
                            # at the end of each epoch
                            validation_data=(data_validation, labels_validation),shuffle=False)
            
                  
            #save history of this batch for epoch i_epoch
            history_ij = history_model.history
            history_i.append(history_ij)
        
            
            #set a threshold for early stopping
            acc = history_ij['binary_accuracy'][0]
            val_acc = history_ij['val_binary_accuracy'][0]
            
            if acc >= threshold and val_acc >= threshold:
                count += 1
            
            #save weights
            filepath = path_dir+'/Model_cnn'+model_choice+'_'+condition+'_E1_'+ init_dir +'-'+str(window_duration)+'epoch'+str(i_epoch)+'_nblocks'+str(n_blocks)+'_'+test+'dropout_factor'+str(dropout_factor)+'_regfactor'+str(regularization_factor)+'.keras'
            model.save(filepath)
            
            #quit loop and end running
            if count > int(n_step*0.95):
                end_running = True
                break
            
         
        #saving the shuffling indexes at the first epoch
        if i_epoch == 0:
            
            #write file for i_epoch = 0:
            #write for the training
            file_name = path_dir_shuffling+'/indexesshuffling_train.npy'
            with open(file_name, 'wb') as file:
                #writing the file
                np.save(file, np.array(index_shuffle_train_global), allow_pickle=True, fix_imports=True)
                file.close()
            
            #write for the validation
            file_name = path_dir_shuffling+'/indexesshuffling_validation.npy'
            with open(file_name, 'wb') as file:
                #writing the file
                np.save(file, np.array(index_shuffle_validation_global), allow_pickle=True, fix_imports=True)
                file.close()
            
        #save history of this epoch
        history_global.append(history_i)
        
        #write file for this epoch
        file_name = path_dir_history+'/history_epoch'+str(i_epoch)+'.npy'
        with open(file_name, 'wb') as file:
            #writing the file
            np.save(file, np.array(history_i), allow_pickle=True, fix_imports=True)
            file.close()
        
        #end running by quitting loop
        if end_running == True:
            break
                
    #write file for the hole epochs  
    file_name = path_dir_history+'/history_global.npy'
    with open(file_name, 'wb') as file:
        #writing the file
        np.save(file, np.array(history_global), allow_pickle=True, fix_imports=True)
        file.close()
            
    #return np.array(history_global)

#input1 = int(input('Write at which step you want to begin : '))
input1 = 0
#input2 = int(input('Write how many epochs you want to do : '))
input2 = 1
#MDC_ornot = input('MDC data or not, type yes or no : ')
MDC_ornot = 'yes'
#threshold = float(input('Write the threshold for early stopping : '))
threshold = 0.80
#input3 = input('Test = yes / No test = no : ')
input3 = 'yes'
#input4 = input('Model choice = simple/Resnet50 : ')
input4 = 'Resnet50'
#input5 = int(input('n_blocks : '))
input5 = 4
#input6 = int(input('number of files : ')) 
input6 = int(1176*0.3)+1
#input7 = input('Write scale_number : ')
input7 = 'scale_2'
#input8 = input('Condition = withnoise/withoutnoise : ')
input8 = 'withnoise'
#input9 = int(input('Size of the window : '))
input9 = 16
#input10 = int(input('Batch size : '))
input10 = 40
#input11 = float(input('Fraction train : '))
input11 = 0.01

history = CNN_compiler(step_0 = input1, n_epoch = input2, MDC_ornot = MDC_ornot, threshold=threshold, test=input3, model_choice = input4, n_blocks=input5, n_files=input6, init_dir=input7, condition=input8, window_duration=input9, batch_size=input10, fraction_train=input11)






#tensorflow data set