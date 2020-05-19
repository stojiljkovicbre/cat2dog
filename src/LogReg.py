import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pywt
import pywt.data
import cv2
import os
import json
import sys

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression

from .utils import (center_image_eyes, set_params, get_params, basic_preprocess_image,
                    basic_preprocess_image_normal, get_features, get_features_normal,
                    load_data, do_SVD, get_accuracy, center_image_eyes, plot_conf_mat, _sample_eyes)

def LogReg(path_list = ['cat2dog/trainA', 'cat2dog/trainB'], num_features = 100, show = True, json_filepath = 'parameters_LogReg.json'):
    """Function that fits Logistic Regression on given data."""
    # Reinitialization
    set_params()
    print('#####################################################################')
    print('Logistic Regression clasifier')
    print('#####################################################################')
    print('Params:')
    print(' number of features: {}'.format(num_features))
    print(' wavelet: {}'.format(get_params()[1]))
    print('#####################################################################')
    # Calculating Singular Value Decomposition
    S, V, DATA, labels, indices, ls, l = do_SVD(path_list)
    # print(S)
    # Choosing number of features original features to be reduced on
    T = V[:, :num_features]
    # Reducing features
    data = T.transpose().dot(DATA)
    # Shuffling the data
    X = data.transpose()
    y = np.array(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Instantiating a model
    model = LogisticRegression(penalty = 'l2',
                               C = 1.0,
                               class_weight = 'balanced',
                               max_iter = 10000,
                               verbose = 0)
    # Fitting the model
    model.fit(X, y)
    # Predicting classes
    y_pred = model.predict(X)
    # Ploting
    if show:
        if num_features == 2:
            t1, t2 = model.coef_[0]
            b = model.intercept_[0]
            f1 = np.linspace(-0.2, 0.2, 100)
            f2 = (-t1 * f1 - b) / t2

            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            scatter1 = ax.scatter(X[:, 0], X[:, 1], c = y)
            plot1 = ax.plot(f1, f2, label="Granica", c = 'red')
            plt.title("Trening podaci")
            plt.xlabel('feature 1')
            plt.ylabel('feature 2')
            # legend1 = ax.legend([*scatter1.legend_elements(), plot1], loc = "upper right", title = "Legend")
            legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Legend")
            ax.add_artist(legend1)
            ax.axis('equal')
            plt.show()

        plot_conf_mat(y_true = y, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    # Saving parameters
    mean, wave, size, f_size = get_params()
    accu = get_accuracy(y_pred = y_pred, y_true = y)
    print(accu)
    model_file = 'model_LogReg.joblib'
    save_params(mean, wave, size, f_size, T, accu, num_features, model_file, model, json_filepath)
    print('#####################################################################')
    return mean, T, accu, num_features

def sample_eyes(path_list = ['cat2dog/testA', 'cat2dog/testB'], json_filepath = 'parameters_LogReg.json'):
    mean, wave, size, f_size, T, accu, num_features, model = load_and_set_params(json_filepath)
    _sample_eyes(path_list, T, num_features)

def test_LogReg(path_list = ['cat2dog/testA', 'cat2dog/testB'], json_filepath = 'parameters_LogReg.json', show = True):
    """Function that tests Logistic Regression classifier on given data."""
    # Loading and setting parameters
    mean, wave, size, f_size, T, accu, num_features, model = load_and_set_params(json_filepath)
    print('#####################################################################')
    print('Testing Linear Discriminant Analysis')
    print('#####################################################################')
    print('Params:')
    print(' number of features: {}'.format(num_features))
    print(' wavelet: {}'.format(get_params()[1]))
    print('#####################################################################')
    # Loading data
    samples, labels, indices, ls, l, DATA = load_data(path_list)
    # Infering
    y_pred = inference_LogReg(DATA, T, model)
    # Calculating accuracy
    y_true = np.array(labels)
    accuracy = get_accuracy(y_pred = y_pred, y_true = y_true)
    # Some basic printing
    print('test accuracy: {:.2f}%'.format(accuracy*100))
    print('Class 0: cats')
    print('Class 1: dogs')
    print('#####################################################################')
    if show:
        # Ploting
        plot_conf_mat(y_true = labels, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    return accuracy

def process_image_given_eyes_LogReg(image, right_eye, left_eye, json_filepath = 'parameters_LogReg.json', features = 'normal'):
    # Loading and setting parameters
    mean, wave, size, f_size, T, accu, num_features, model = load_and_set_params(json_filepath)
    # Image centering
    image = center_image_eyes(image, right_eye, left_eye)
    # Gettig normal features
    features = get_features_normal(image)
    # Infering
    y_pred = inference_LogReg(features, T, model)[0]
    return y_pred, image

def inference_LogReg(X, T, model):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis = 1)
    X = T.transpose().dot(X)
    X = X.transpose()
    y_pred = model.predict(X)
    return y_pred

def load_and_set_params(json_filepath):
    # Loading parameters
    with open(json_filepath) as f:
        data = json.load(f)
    mean = np.array(data['mean'])
    wave = data['wavelet']
    size = data['size']
    f_size = data['f_size']
    T = np.array(data['T'])
    accu = np.array(data['accu'])
    num_features = data['num_features']
    model_file = data['model_file']
    # Setting parameters
    set_params(m = mean, wave = wave, s = size, f_s = f_size)
    model = joblib.load(model_file)
    return mean, wave, size, f_size, T, accu, num_features, model

def save_params(mean, wave, size, f_size, T, accu, num_features, model_file, model, json_filepath):
    json_data = {'mean': mean.tolist(), 'wavelet': wave, 'size': size, 'f_size': f_size,
                 'T': T.tolist(), 'accu': accu, 'num_features': num_features,
                 'model_file': model_file}
    print('Saving data...')
    with open(json_filepath, 'w') as f:
        json.dump(json_data, f)
    joblib.dump(model, model_file)
    print('Data saved')
