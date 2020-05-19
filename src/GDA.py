import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pywt
import pywt.data
import cv2
import os
import json
import sys
import time

from scipy.stats import multivariate_normal

from .utils import (center_image_eyes, set_params, get_params, basic_preprocess_image,
                    basic_preprocess_image_normal, get_features, get_features_normal,
                    load_data, do_SVD, get_accuracy, center_image_eyes, plot_conf_mat, _sample_eyes)

def temp_finction(path_list = ['cat2dog/trainA', 'cat2dog/trainB'], num_features = 100, show = True, json_filepath = 'parameters_GDA.json'):
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
    return S, V, DATA, labels, indices, ls, l

def sample_eyes(path_list = ['cat2dog/testA', 'cat2dog/testB'], json_filepath = 'parameters_GDA.json'):
    mean, T, P, M, COV, accu, num_features, wave = load_and_set_params(json_filepath)
    _sample_eyes(path_list, T, num_features)

def LDA(path_list = ['cat2dog/trainA', 'cat2dog/trainB'], num_features = 100, show = True, json_filepath = 'parameters_GDA.json'):
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
    # plt.plot(np.arange(len(S)), S)
    # plt.xlabel('N')
    # plt.title('Sopstvene vredonsti')
    # plt.show()
    # print(S)
    # print('max', np.max(DATA))
    # print('min', np.min(DATA))
    # Choosing number of features original features to be reduced on
    T = V[:, :num_features]
    # Reducing features
    cats = T.transpose().dot(DATA[:, indices[0]])
    dogs = T.transpose().dot(DATA[:, indices[1]])
    data = T.transpose().dot(DATA)
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
    # Packing parameters
    P = np.array([P_cat, P_dog])
    M = np.array([M_cat, M_dog])
    COV = np.array([S_cat, S_dog])
    # Inferings
    y_pred = inference_LDA(DATA, T, P, M, COV)
    accu = get_accuracy(y_pred = y_pred, y_true = labels)
    print('train accuracy: {:.2f}%'.format(accu*100))
    print('Class 0: cats')
    print('Class 1: dogs')
    if show:
        # Ploting
        if num_features == 2:

            delta = 0.02
            x1_range = np.arange(-1, 1, delta)
            x2_range = np.arange(-1, 1, delta)
            X1, X2 = np.meshgrid(x1_range, x2_range)

            q_mat = np.linalg.inv(S_dog) - np.linalg.inv(S_cat)
            l_mat = M_dog.transpose().dot(np.linalg.inv(S_dog)) - M_cat.transpose().dot(np.linalg.inv(S_cat))
            l_mat = np.squeeze(l_mat)
            B = np.linalg.det(S_dog) - np.linalg.det(S_cat) + M_dog.transpose().dot(np.linalg.inv(S_dog)).dot(M_dog) - M_cat.transpose().dot(np.linalg.inv(S_cat)).dot(M_cat)

            q1 = q_mat[0, 0]
            q12 = q_mat[0, 1] + q_mat[1, 0]
            q2 = q_mat[1, 1]

            p1 = l_mat[0]
            p2 = l_mat[1]

            F = q1*X1*X1 + q12*X1*X2 + q2*X2*X2 - 2*p1*X1 - 2*p2*X2 + 2* np.log(P_cat / P_dog) + B

            classes = np.array(labels)
            ax = plt.gca()
            scatter1 = ax.scatter(data[0, :], data[1, :], c = classes)
            ax.contour(X1, X2, F, [0], colors = ['red'])
            plt.title("Trening podaci")
            plt.xlabel('feature 1')
            plt.ylabel('feature 2')
            legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Class")
            ax.add_artist(legend1)
            ax.axis('equal')
            plt.show()
        # Plot confusion matrix
        plot_conf_mat(y_true = labels, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    # Saving final parameters
    mean, wave, size, f_size = get_params()
    save_params(mean, wave, size, f_size, T, P, M, COV, accu, num_features, json_filepath)
    print('#####################################################################')
    return accu

def test_LDA(path_list = ['cat2dog/testA', 'cat2dog/testB'], json_filepath = 'parameters_GDA.json', show = True):
    """Function that tests LDA algorithm on given data."""
    # Loading and setting parameters
    mean, T, P, M, COV, accu, num_features, wave = load_and_set_params(json_filepath)
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
    y_pred = inference_LDA(DATA, T, P, M, COV)
    # Calculating accuracy
    accuracy = get_accuracy(y_pred = y_pred, y_true = labels)
    # Some basic printing
    print('test accuracy: {:.2f}%'.format(accuracy*100))
    print('Class 0: cats')
    print('Class 1: dogs')
    print('#####################################################################')
    if show:
        # Ploting
        if num_features == 2:
            
            P_cat, P_dog = P
            M_cat, M_dog = M
            S_cat, S_dog = COV
            data = T.transpose().dot(DATA)

            delta = 0.02
            x1_range = np.arange(-1, 1, delta)
            x2_range = np.arange(-1, 1, delta)
            X1, X2 = np.meshgrid(x1_range, x2_range)

            q_mat = np.linalg.inv(S_dog) - np.linalg.inv(S_cat)
            l_mat = M_dog.transpose().dot(np.linalg.inv(S_dog)) - M_cat.transpose().dot(np.linalg.inv(S_cat))
            l_mat = np.squeeze(l_mat)
            B = np.linalg.det(S_dog) - np.linalg.det(S_cat) + M_dog.transpose().dot(np.linalg.inv(S_dog)).dot(M_dog) - M_cat.transpose().dot(np.linalg.inv(S_cat)).dot(M_cat)

            q1 = q_mat[0, 0]
            q12 = q_mat[0, 1] + q_mat[1, 0]
            q2 = q_mat[1, 1]

            p1 = l_mat[0]
            p2 = l_mat[1]

            F = q1*X1*X1 + q12*X1*X2 + q2*X2*X2 - 2*p1*X1 - 2*p2*X2 + 2* np.log(P_cat / P_dog) + B

            classes = np.array(labels)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            scatter1 = ax.scatter(data[0, :], data[1, :], c = classes)
            ax.contour(X1, X2, F, [0], colors = ['red'])
            plt.title("Trening podaci")
            plt.xlabel('feature 1')
            plt.ylabel('feature 2')
            legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Class")
            ax.add_artist(legend1)
            ax.axis('equal')
            plt.show()
        # Plot confusion matrix
        plot_conf_mat(y_true = labels, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')
    return accuracy

def process_image_given_eyes_LDA(image, right_eye, left_eye, json_filepath = 'parameters_GDA.json', features = 'normal'):
    start = time.time()
    # Loading and setting parameters
    mean, T, P, M, COV, accu, num_features, wave = load_and_set_params(json_filepath)
    # Image centering
    image = center_image_eyes(image, right_eye, left_eye) # uint8
    # Getting normal features
    features = get_features_normal(image)
    # Infering
    y_pred = inference_LDA(features, T, P, M, COV)[0]
    # print('image processed in {:.3f}'.format(time.time() - start))
    return y_pred, image

def inference_LDA(X, T, P, M, COV):
    assert len(M) == len(COV) and len(COV) == len(P), 'ERROR'
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis = 1)
    X = T.transpose().dot(X)
    X = X.transpose()
    Q = np.zeros((len(M), X.shape[0]))
    for j, p, m, cov in zip(range(len(M)), P, M, COV):
        m = np.squeeze(m)
        q = np.zeros((1, X.shape[0]))
        for i, x in enumerate(X):
            q[0, i] = p * multivariate_normal.pdf(x, mean = m, cov = cov)
        Q[j, :] = q
    y_pred = np.argmax(Q, axis = 0)
    return y_pred

def load_and_set_params(json_filepath = 'parameters_GDA.json'):
    # Loading parameters
    with open(json_filepath) as f:
        data = json.load(f)
    mean = np.array(data['mean'])
    wave = data['wavelet']
    size = data['size']
    f_size = data['f_size']
    T = np.array(data['T'])
    P = np.array(data['P'])
    M = np.array(data['M'])
    COV = np.array(data['COV'])
    accu = data['accu']
    num_features = data['num_features']
    # Setting parameters
    set_params(m = mean, wave = wave, s = size, f_s = f_size)
    return mean, T, P, M, COV, accu, num_features, wave

def save_params(mean, wave, size, f_size, T, P, M, COV, accu, num_features, json_filepath):
    json_data = {'mean': mean.tolist(), 'wavelet': wave, 'size': size, 'f_size': f_size,
                 'T': T.tolist(), 'P': P.tolist(), 'M': M.tolist(), 'COV': COV.tolist(), 'accu': accu,
                 'num_features': num_features}
    print('Saving data...')
    with open(json_filepath, 'w') as f:
        json.dump(json_data, f)
    print('Data saved')
