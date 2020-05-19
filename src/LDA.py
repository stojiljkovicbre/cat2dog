import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pywt
import pywt.data
import cv2
import os
import json
import sys

from .utils import (center_image_eyes, set_params, get_params, basic_preprocess_image,
                    basic_preprocess_image_normal, get_features, get_features_normal,
                    load_data, do_SVD, get_accuracy, center_image_eyes, plot_conf_mat)

def LDA(path_list = ['cat2dog/trainA', 'cat2dog/trainB'], num_features = 100, show = True, json_filepath = 'parameters_LDA.json'):
    """Function that does Linear Discriminant Analysis (LDA) on given data."""
    # Reinitialization
    set_params()
    print('#####################################################################')
    print('Linear Discriminant Analysis')
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
    cats = T.transpose().dot(DATA[:, indices[0]])
    dogs = T.transpose().dot(DATA[:, indices[1]])
    # Data mean
    M = np.expand_dims(np.mean(T.transpose().dot(DATA), axis = 1), axis = 1)
    # Class 0 mean
    M_cat = np.expand_dims(np.mean(cats, axis = 1), axis = 1)
    # Class 1 mean
    M_dog = np.expand_dims(np.mean(dogs, axis = 1), axis = 1)
    # Extracting dimensions
    l_f, l_cat = cats.shape
    l_f, l_dog = dogs.shape
    # Calculating class 0 scatter matrix
    S_cat = np.zeros((l_f, l_f))
    for i in range(l_cat):
        cat = np.expand_dims(cats[:, i], axis = 1)
        S_cat = S_cat + (cat - M_cat).dot((cat - M_cat).transpose())
    S_cat = S_cat / l_cat
    # Calculating class 1 scatter matrix
    S_dog = np.zeros((l_f, l_f))
    for i in range(l_dog):
        dog = np.expand_dims(dogs[:, i], axis = 1)
        S_dog = S_dog + (dog - M_dog).dot((dog - M_dog).transpose())
    S_dog = S_dog / l_dog
    # Calculating class probabilities
    P_cat = l_cat / (l_cat + l_dog)
    P_dog = l_dog / (l_cat + l_dog)
    # Calculating within-class scatter matrix
    Sw = P_cat*S_cat + P_dog*S_dog
    # Calculating between-class scatter matrix
    Sb = P_cat*(M_cat - M).dot((M_cat - M).transpose()) + P_dog*(M_dog - M).dot((M_dog - M).transpose())
    # Calculating criteria
    J = np.linalg.inv(Sw).dot(Sb)
    # Solving for given criteria
    d, v = np.linalg.eig(J)
    # Finding vector w, normal on separating hypreplane
    ind = np.argmax(d)
    w = np.real(v[:, ind])
    w = w / np.linalg.norm(w)
    w = np.expand_dims(w, axis = 1)
    # Projecting data on vector w
    cats = w.transpose().dot(cats)
    dogs = w.transpose().dot(dogs)
    # Concatenating cats and dogs
    data = np.squeeze(np.concatenate((cats, dogs), axis = 1))
    # Initalization for finding optimal threshold
    t_range = np.linspace(np.min([np.min(cats), np.min(dogs)]), np.max([np.max(cats), np.max(dogs)]), 5000)
    t_opt = t_range[0]
    accu_max = 0
    print('Finding optimal threshold value...')
    for i, t in enumerate(t_range):
        # y_pred = inference_LDA(DATA, T, w, t)
        accu = classify_LDA(data, labels, thresh = t)
        # accu = get_accuracy(y_pred, labels)
        if accu > accu_max:
            t_opt = t
            accu_max = accu
        print('progress: {}%'.format(int((i + 1)/len(t_range) * 100)), end = '\r')
    print('\n')
    # Some basic printing
    print('optimal threshold value found: ', t_opt)
    print('train accuracy: {:.2f}%'.format(accu_max*100))
    print('Class 0: cats')
    print('Class 1: dogs')
    if show:
        # Ploting
        classes = np.array(labels)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        scatter1 = ax.scatter(data, np.zeros((l,)), c = classes)
        # ax.scatter(np.squeeze(cats), np.zeros((l_cat,)), c = 'red')
        # ax.scatter(np.squeeze(dogs), np.zeros((l_dog,)), c = 'blue')
        l = mlines.Line2D([t_opt, t_opt], [-0.005, 0.005])
        ax.add_line(l)
        plt.title("Treting podaci: projekcija klasa 'cat' i 'dog' na vektor w")
        plt.xlabel('projekcija na vektor w')
        legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Class")
        ax.add_artist(legend1)
        plt.savefig('train_data_wproj_plot.png')
        plt.show()
        # Plot confusion matrix
        y_pred = inference_LDA(DATA, T, w, t_opt)
        plot_conf_mat(y_true = labels, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    # Saving final parameters
    mean, wave, size, f_size = get_params()
    save_params(mean, wave, size, f_size, T, w ,t_opt, accu_max, num_features, json_filepath)
    print('#####################################################################')
    return mean, T, w, t_opt, accu_max

def test_LDA(path_list = ['cat2dog/testA', 'cat2dog/testB'], json_filepath = 'parameters_LDA.json', show = True):
    """Function that tests LDA algorithm on given data."""
    # Loading and setting parameters
    mean, T, w, t_opt, accu_max, num_features, wave = load_and_set_params(json_filepath)
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
    y_pred = inference_LDA(DATA, T, w, t_opt)
    # Calculating accuracy
    accuracy = get_accuracy(y_pred = y_pred, y_true = labels)
    # Calculating projection of samples on the vector normal to separating line
    data = LDA_projection(DATA, T, w)
    # Some basic printing
    print('test accuracy: {:.2f}%'.format(accuracy*100))
    print('Class 0: cats')
    print('Class 1: dogs')
    print('#####################################################################')
    if show:
        # Ploting
        classes = np.array(labels)
        ax = plt.gca()
        scatter1 = ax.scatter(data, np.zeros((l,)), c = classes)
        l = mlines.Line2D([t_opt, t_opt], [-0.005, 0.005])
        ax.add_line(l)
        plt.title("Test podaci: projekcija klasa 'cat' i 'dog' na vektor w")
        plt.xlabel('projekcija na vektor w')
        ax.set_xticks([])
        ax.set_yticks([])
        legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Class")
        ax.add_artist(legend1)
        plt.savefig('test_data_wproj_plot.png')
        plt.show()
        # Plot confusion matrix
        plot_conf_mat(y_true = labels, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    return accuracy

def process_image_given_eyes_LDA(image, right_eye, left_eye, json_filepath = 'parameters_LDA.json', features = 'normal'):
    # Loading and setting parameters
    mean, T, w, t_opt, accu_max, num_features, wave = load_and_set_params(json_filepath)
    # Image centering
    image = center_image_eyes(image, right_eye, left_eye)
    # Gettig normal features
    features = get_features_normal(image)
    # Infering
    y_pred = inference_LDA(features, T, w, t_opt)[0]
    return y_pred, image

def load_and_set_params(json_filepath):
    # Loading parameters
    with open(json_filepath) as f:
        data = json.load(f)
    mean = np.array(data['mean'])
    wave = data['wavelet']
    size = data['size']
    f_size = data['f_size']
    T = np.array(data['T'])
    w = np.array(data['w'])
    t_opt = data['t_opt']
    accu_max = data['accu_max']
    num_features = data['num_features']
    # Setting parameters
    set_params(m = mean, wave = wave, s = size, f_s = f_size)
    return mean, T, w, t_opt, accu_max, num_features, wave

def save_params(mean, wave, size, f_size, T, w ,t_opt, accu_max, num_features, json_filepath):
    json_data = {'mean': mean.tolist(), 'wavelet': wave, 'size': size, 'f_size': f_size,
                 'T': T.tolist(), 'w': w.tolist(), 't_opt': t_opt, 'accu_max': accu_max,
                 'num_features': num_features}
    print('Saving data...')
    with open(json_filepath, 'w') as f:
        json.dump(json_data, f)
    print('Data saved')

def classify_LDA(data, labels, thresh):
    pred = []
    data_s = np.squeeze(data)
    for d in data_s:
        if d <= thresh:
            pred.append(0)
        else:
            pred.append(1)
    sum = 0
    for p, l in zip(pred, labels):
        if p == l:
            sum += 1
    return sum / len(labels)

def LDA_projection(X, T, w):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis = 1)
    X = X.transpose()
    projections = []
    for x in X:
        x = T.transpose().dot(x)
        x = w.transpose().dot(x)
        projections.append(x)
    return projections

def inference_LDA(X, T, w, t_opt):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis = 1)
    X = X.transpose()
    y_pred = []
    for x in X:
        x = LDA_projection(x, T, w)[0]
        if x <= t_opt:
            y_pred.append(0)
        else:
            y_pred.append(1)
    # return np.array(y_pred)
    return y_pred
