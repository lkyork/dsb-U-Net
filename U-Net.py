#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:39:39 2018

@author: yorklk
"""


import os
import numpy as np
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input, Activation, Add, BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
import keras.backend as KB
import tensorflow as tf


X_train = np.load('data/X_128.npy')
Y_train = np.load('data/Y_128.npy')
X_test = np.load('data/test_128.npy')

np.random.seed(seed = 71)
epochs = 500
learning_rate = 1e-3
learning_rates = [1e-3]
decay = 5e-5
patience = 15
dropout_rate = 0.015
batch_size = 8
K=4
pre_proc = dict(horizontal_flip = True,
                vertical_flip = True,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                channel_shift_range= 0.0,
                zoom_range = 0.0,
                rotation_range = 0.0)

####################################################################

# Precision helper function
def precision_at(threshold, iou):

    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   
    false_positives = np.sum(matches, axis=0) == 0  
    false_negatives = np.sum(matches, axis=1) == 0  
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    return tp, fp, fn


def iou_metric(y_true_in, y_pred_in):

    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas 
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)

    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):

    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)

    return np.array(np.mean(metric), dtype=np.float32)


def my_iou_metric(label, pred):

    metric_value = iou_metric_batch(label, pred)

    return metric_value

#########################################################################################################


def identity_block(X, filters, dropout_rate):

    F1, F2, F3 = filters
    
    # First component 
    X = Conv2D(F1, kernel_size = (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)
    
    # Second component
    X = Conv2D(F2, (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)
    
    # Third component
    X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)

    return X


def deconvolution_block(X, filters, dropout_rate):

    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component, deconvolution
    X = Conv2DTranspose(F1,
                        (2, 2),
                        strides=(2, 2),
                        padding='same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)
    
    # Second component
    X = Conv2D(F2, (3, 3), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)
    
    # Third component 
    X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    
    # Shortcut deconvolution
    X_shortcut = Conv2DTranspose(F3,
                                 (2, 2),
                                 strides=(2, 2),
                                 padding='same')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
    
    # Add shortcut value to main path, and ELU activation
    X = Add()([X_shortcut, X])
    X = Activation('elu')(X)
    X = Dropout(rate = dropout_rate)(X)

    return X


def uNet_Model(input_shape = (128, 128, 3), dropout_rate = dropout_rate):

    '''
    uNet with MobileNet (pretrained on imagenet) as downsampling side
    Outputs saved from five layers to concatenate on upsampling side:
    activations at conv_pw_1, conv_pw_3, conv_pw_5, conv_pw_11, conv_pw_13
    ResNet convolution blocks with Conv2DTranspose and elu used for upsampling side
    '''
    
    base_model = MobileNet(weights = 'imagenet',
                           include_top = False,
                           input_shape = (128, 128, 3))
    
    # Base model, with 5 layers out
    X1 = base_model.get_layer('conv_pw_1_relu').output # 64x64, 64 filters
    X2 = base_model.get_layer('conv_pw_3_relu').output # 32x32, 128 filters
    X3 = base_model.get_layer('conv_pw_5_relu').output # 16x16, 256 filters
    X4 = base_model.get_layer('conv_pw_11_relu').output # 8x8, 512 filters
    X5 = base_model.get_layer('conv_pw_13_relu').output # 4x4, 1024 filters
    
    # Bottom block
    X = identity_block(X5, filters = [256, 256, 1024], dropout_rate = dropout_rate) # 4x4
    X = Add()([X, X5]) # 4x4
    
    # Deconvolution block 1
    X = deconvolution_block(X, filters = [128, 128, 512], dropout_rate = dropout_rate) # 8x8
    X = Add()([X, X4])  # 8x8
    
    # Deconvolution block 2
    X = deconvolution_block(X, filters = [64, 64, 256], dropout_rate = dropout_rate) # 16x16
    X = Add()([X, X3]) # 16x16
    
    # Deconvolution block 3
    X = deconvolution_block(X, filters= [32, 32, 128], dropout_rate = dropout_rate) # 32x32
    X = Add()([X, X2]) # 32x32
    
    # Deconvolution block 4
    X = deconvolution_block(X, filters = [16, 16, 64], dropout_rate = dropout_rate) # 64x64
    X = Add()([X, X1]) # 64x64

    # Final deconvolution block
    X = deconvolution_block(X, filters = [16, 16, 64], dropout_rate = dropout_rate) # 128x128

    predictions = Conv2D(1, (1, 1), activation='sigmoid')(X)

    model = Model(input = base_model.input, output = predictions)

    return model


#####################################################################################################


def train_uNet(X_train_cv, Y_train_cv, X_dev, Y_dev, parameters, batch_size, train_generator, file_path):

    # Train model using Adam optimizer and early stopping
    model = uNet_Model(input_shape=(128, 128, 3), dropout_rate = parameters['dropout_rate'])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr = parameters['learning_rate'], decay = parameters['decay']),
                  metrics=['accuracy'])
    model.fit_generator(generator = train_generator,
                        steps_per_epoch = int(X_train_cv.shape[0]/batch_size),
                        epochs = epochs,
                        shuffle = True,
                        verbose = 2,
                        validation_data = (X_dev, Y_dev),
                        validation_steps = int(X_train_cv.shape[0]/batch_size),
                        callbacks = [EarlyStopping('val_loss', patience=parameters['patience'], mode="min"),
                                     ModelCheckpoint(file_path, save_best_only=True)])

    return model


def get_folds(X_train, Y_train, K):

    # Shuffles data then returns K folds of X,Y-train, X,Y-dev
    folds = []
    m = X_train.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffled = X_train[permutation, :, :, :]
    Y_shuffled = Y_train[permutation, :, :, :]
    fold_length = int(m/K)
    for j in range(K):
        cv_idx = list(range(j*fold_length, (j+1)*fold_length))
        train_idx = list(range(0, j*fold_length)) + list(range((j+1)*fold_length, m))
        X_train_cv = X_shuffled[train_idx, :, :, :]
        Y_train_cv = Y_shuffled[train_idx, :, :, :]
        X_dev = X_shuffled[cv_idx, :, :, :]
        Y_dev = Y_shuffled[cv_idx, :, :, :]
        fold = [X_train_cv, Y_train_cv, X_dev, Y_dev]
        folds.append(fold)

    return folds


def get_file_path(j, parameters, directory):

    print('\nFold:\t{}\nlearning_rate:\t{learning_rate}\ndropout_rate:\t{dropout_rate}\naugmentation:\t{aug}'.format(str(j), **parameters))
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = directory + '/weights_'+str(j)+'.hdf5'

    return file_path


def rename_weight_path(j, metrics, file_path, directory):

    print('\nFold:\t{}\nTrain Loss:\t{train_loss:.4}\nDev Loss:\t{dev_loss:.4}\nMean IoU:\t{IoU:.4}\n'.format(str(j), **metrics))
    new_weight_path = '{}_{}_{dev_loss:.4}_{IoU:.4}{hdf5}'.format('/weights', str(j), **metrics)
    os.rename(file_path, directory + new_weight_path)

    return


def print_final_metrics(parameters, metrics, directory):

    print('\n\nlearning_rate: {learning_rate}\ndropout_rate: {dropout_rate}\naugmentation: {aug}'.format(**parameters))
    print('avg_dev_loss:\t{avg_dev_loss}\nmean_IoU:\t{IoU_log}\n\n\n'.format(**metrics))
    name = 'loss={avg_dev_loss:.6}_IoU={IoU_log:.6}'.format(**metrics)
    new_path = directory+'--'+name
    os.rename(directory, new_path)

    return


def get_metrics(model, X_train_cv, Y_train_cv, X_dev, Y_dev, file_path, metrics):

    # Load the best model weights saved by early stopping
    K = metrics['K']
    model.load_weights(filepath=file_path)
    
    # Get train and dev loss
    train_eval = model.evaluate(X_train_cv, Y_train_cv, verbose=0)
    metrics['train_loss'] = train_eval[0]
    dev_eval = model.evaluate(X_dev, Y_dev, verbose=0)
    metrics['dev_loss'] = dev_eval[0]
    metrics['avg_dev_loss'] += metrics['dev_loss']/K
    
    # Get Intersection over Union
    preds_dev = model.predict(X_dev)
    Y_pred = preds_dev >= 0.5
    Y_true = Y_dev
    IoU = my_iou_metric(Y_true, Y_pred)
    metrics['IoU'] = IoU
    metrics['IoU_log'] += IoU/K

    return metrics

############################################################################

for learning_rate in learning_rates:

    aug = ''
    if pre_proc['width_shift_range'] != 0.0:
        aug += '_{}={width_shift_range}'.format(aug, 'shift', **pre_proc)
    if pre_proc['zoom_range'] != 0.0:
        aug += '_{}={zoom_range}'.format(aug, 'zoom', **pre_proc)
    if pre_proc['rotation_range'] != 0.0:
        aug += '_{}={rotation_range}'.format(aug, 'rotation', **pre_proc)
    if pre_proc['horizontal_flip']:
        aug += '_{}={horizontal_flip}'.format(aug, 'h-flip', **pre_proc)
    if pre_proc['vertical_flip']:
        aug += '_{}={vertical_flip}'.format(aug, 'v-flip', **pre_proc)

    parameters = {'learning_rate':learning_rate, 'dropout_rate':dropout_rate, 'aug':aug, 'decay':decay, 'patience':patience}
    directory = 'model_5/{learning_rate}_{dropout_rate}/{aug}'.format(**parameters)
    metrics = {'train_loss': 0, 'dev_loss': 0, 'avg_dev_loss': 0, 'IoU': 0, 'IoU_log': 0, 'K': K,
               'hdf5': '.hdf5'}

    # Create image and mask data generators for preprocessing
    image_datagen = ImageDataGenerator(**pre_proc)
    mask_datagen = ImageDataGenerator(**pre_proc)
    image_datagen.fit(X_train, augment = True)
    mask_datagen.fit(Y_train, augment = True)
    image_generator = image_datagen.flow(X_train,
                                         batch_size = batch_size)
    mask_generator = mask_datagen.flow(Y_train,
                                       batch_size = batch_size)
    train_generator = zip(image_generator, mask_generator)
    
    # Create folds and train
    folds = get_folds(X_train, Y_train, K)
    for j in range(K):
        X_train_cv = folds[j][0]
        Y_train_cv = folds[j][1]
        X_dev = folds[j][2]
        Y_dev = folds[j][3]
        file_path = get_file_path(j, parameters, directory)
        model = train_uNet(X_train_cv, Y_train_cv, X_dev, Y_dev, parameters, batch_size,
                           train_generator, file_path)
        metrics = get_metrics(model, X_train_cv, Y_train_cv, X_dev, Y_dev, file_path, metrics)
        rename_weight_path(j, metrics, file_path, directory)
    print_final_metrics(parameters, metrics, directory)

