import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pywt
import pywt.data
import cv2
import os
import json
import sys
import itertools
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

size = 128
f_size = 64
wavelet = 'haar'
mean = np.array(None)

def set_params(s = 128, f_s = 64, wave = 'haar', m = np.array(None)):
    global size, f_size, mean, wavelet
    size = s
    f_size = f_s
    mean = m
    wavelet = wave

def get_params():
    global size, f_size, mean, wavelet
    return (mean, wavelet, size, f_size)

def basic_preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img[24:-24, 4:-4], (size, size))
    img = normalize_range(img)
    return img

def basic_preprocess_image_normal(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    img = normalize_range(img)
    return img

def normalize_range(image):
    img = image - np.min(image)
    img = img / np.max(img)
    return img

def DWT(image, filter):
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    img = np.vstack((np.hstack((LL, HL)), np.hstack((LH, HH))))
    img = normalize_range(img)
    return img, LL, (LH, HL, HH)

def get_features(image):
    global mean
    img = basic_preprocess_image(image)
    LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
    x = (LH + HL) / 2
    if np.equal(mean, None).any():
        return np.reshape(x, (f_size*f_size,))
    else:
        return np.reshape(x, (f_size*f_size,)) - mean

def get_features_normal(image):
    global mean
    img = basic_preprocess_image_normal(image)
    LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
    x = (LH + HL) / 2
    if np.equal(mean, None).any():
        return np.reshape(x, (f_size*f_size,))
    else:
        return np.reshape(x, (f_size*f_size,)) - mean

def load_data(path_list):
    """Function that loads the data"""
    samples = []
    labels = []
    indices = []
    ls = []
    index = 0
    # Reding given data names
    for i, path in enumerate(path_list):
        lines = os.listdir(path)
        ls.append(len(lines))
        ind = ()
        for line in lines:
            samples.append(path + '/' + line)
            labels.append(i)
            ind = ind + (index,)
            index = index + 1
        indices.append(ind)
    l = np.sum(ls)
    # Loading the data
    DATA = np.zeros((f_size*f_size, l))
    for i, sample in enumerate(samples):
        image = cv2.imread(sample)
        # Getting features
        features = get_features(image)
        DATA[:, i] = features
        print('loading images: {}/{}'.format(i + 1, l), end = '\r')
    print('\n')
    return samples, labels, indices, ls, l, DATA

def do_SVD(path_list):
    """Function that does Singular Value Decomposition (SVD) on given data."""
    global mean
    # Loading data
    samples, labels, indices, ls, l, DATA = load_data(path_list)
    if np.equal(mean, None).any():
        # Calculating data's mean
        set_params(wave = get_params()[1], m = np.mean(DATA, axis = 1))
        # set_mean(np.mean(DATA, axis = 1))
        # Centering data
        for i in range(DATA.shape[1]):
            DATA[:, i] = DATA[:, i] - mean
            print('centering images: {}/{}'.format(i + 1, l), end = '\r')
        print('\n')
    print('Calculating SVD...')
    # calucating SVD
    u, s, vh = np.linalg.svd(DATA)
    print('SVD done')
    return s, u[:, :vh.shape[1]], DATA, labels, indices, ls, l

def get_accuracy(y_pred, y_true):
    assert len(y_pred) == len(y_true), 'y_pred and y_true must be of same length: len(y_pred): {}, len(y_true): {}'.format(len(y_pred), len(y_true))
    sum = 0
    for p, t in zip(y_pred, y_true):
        if p == t:
            sum += 1
    return sum / len(y_true)

# def plot_conf_mat(y_true, y_pred, normalize):
    # # import some data to play with
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # class_names = iris.target_names
    # print(type(class_names))
    # print(class_names)
    #
    # # Split the data into a training set and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #
    # # Run classifier, using a model that is too regularized (C too low) to see
    # # the impact on the results
    # classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
    #
    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #                   ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     c_mat = confusion_matrix(classifier, X_test, y_test,
    #                                  display_labels=class_names,
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp = plot_confusion_matrix(classifier, X_test, y_test,
    #                                  display_labels=class_names,
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp.ax_.set_title(title)
    #
    #     print(title)
    #     print(disp.confusion_matrix)
    #
    # plt.show()

def plot_conf_mat(y_true, y_pred, classes, normalize = None,
                          title = 'Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize = 'true'`.
    """
    clases = np.array(classes)
    cm = confusion_matrix(y_true, y_pred, normalize = normalize)
    if normalize:
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return cm

def center_image_eyes(image, right_eye, left_eye):
    a, b, _ = image.shape
    y1, x1 = right_eye
    y2, x2 = left_eye
    theta = np.arctan((y2 - y1) / (x2 - x1))
    c = np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))
    xc = int(x1 + c/2 * np.cos(theta))
    yc = int(y1 + c/2 * np.sin(theta))
    theta = int(theta / np.pi *180)
    scale = 33 / c
    M = cv2.getRotationMatrix2D((xc, yc), theta, scale)
    image = cv2.warpAffine(image, M, (b, a))
    image = image[yc-65:yc+128-65, xc-62:xc+128-62, :] # image of type uint8
    return image

def _sample_eyes(path_list, T, num_features):
    rightEye = np.array((110, 67))
    leftEye = np.array((109, 109))
    cov = np.eye(2)*80

    if num_features == 2:
        # Loading data
        samples, labels, indices, ls, l, DATA = load_data(path_list)
        data = T.transpose().dot(DATA)
        X = data.transpose()
        y = np.array(labels)

        cat_path, dog_path = ['cat2dog/testA', 'cat2dog/testB']

        for cat in os.listdir(cat_path):
            # Reading image
            image = cv2.imread(cat_path + '/' + cat)
            for i in range(10):
                # Sampling eyes
                right_eye = np.random.multivariate_normal(rightEye, cov).astype(np.int32)
                left_eye = np.random.multivariate_normal(leftEye, cov).astype(np.int32)
                # Image centering
                image_temp = center_image_eyes(image, right_eye, left_eye)
                # Gettig normal features
                features = get_features_normal(image_temp)
                # Lowering dimension
                x = T.transpose().dot(features)
                X = np.vstack((X, x))
                y = np.append(y, 2)

        for dog in os.listdir(dog_path):
            # Reading image
            image = cv2.imread(dog_path + '/' + dog)
            # cv2.imshow('demo', image)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     exit()
            for i in range(10):
                # Sampling eyes
                right_eye = np.random.multivariate_normal(rightEye, cov).astype(np.int32)
                left_eye = np.random.multivariate_normal(leftEye, cov).astype(np.int32)
                # Image centering
                image_temp = center_image_eyes(image, right_eye, left_eye)
                # Gettig normal features
                features = get_features_normal(image_temp)
                # Lowering dimension
                x = T.transpose().dot(features)
                X = np.vstack((X, x))
                y = np.append(y, 3)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        scatter1 = ax.scatter(X[:, 0], X[:, 1], c = y)
        plt.title("Trening podaci")
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        # legend1 = ax.legend([*scatter1.legend_elements(), plot1], loc = "upper right", title = "Legend")
        legend1 = ax.legend(*scatter1.legend_elements(), loc = "upper right", title = "Legend")
        ax.add_artist(legend1)
        ax.axis('equal')
        plt.show()
    else:
        print('Number of features different than 2.')


# def main():
#     set_params()
#     print(mean)

if __name__ == '__main__':
    main()
