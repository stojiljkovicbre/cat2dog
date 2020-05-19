from src.GDA import LDA, process_image_given_eyes_LDA
from src.utils import get_accuracy, plot_conf_mat
import os
import numpy as np
import cv2
import copy

rightEye = np.array((110, 67))
leftEye = np.array((109, 109))
cov = np.eye(2)*70

def test_robustness(path_list):

    num_classes = len(path_list)
    classes = range(num_classes)
    y_ground = []
    names = []
    for i, path in enumerate(path_list):
        for name in os.listdir(path):
            names.append('{}/{}'.format(path, name))
            y_ground.append(classes[i])

    # print('num_classes', num_classes)
    # print('classes', classes)
    # print('y_grond', y_ground)
    # print('names', names)

    y_true = []
    y_pred = []
    num_samples = 100

    for k, name, label in zip(range(len(names)), names, y_ground):
        image = cv2.imread(name)
        img = copy.deepcopy(image)
        for i in range(num_samples):
            right_eye = np.random.multivariate_normal(rightEye, cov).astype(np.int32)
            left_eye = np.random.multivariate_normal(leftEye, cov).astype(np.int32)
            y, _ = process_image_given_eyes_LDA(img, right_eye, left_eye)
            y_true.append(label)
            y_pred.append(y)
            s = 1
            image[right_eye[0]-s:right_eye[0]+s, right_eye[1]-s:right_eye[1]+s, :] = np.array([255, 255, 0])
            image[left_eye[0]-s:left_eye[0]+s, left_eye[1]-s:left_eye[1]+s, :] = np.array([0, 255, 255])
            cv2.imshow('demo', image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
            elif key == ord('p'):
                key == cv2.waitKey()
            else:
                pass
        print('progress: {}/{}'.format(k+1, len(names)), end = '\r')

    accu = get_accuracy(y_pred = y_pred, y_true = y_true)
    print('accuracy: {}'.format(accu))
    plot_conf_mat(y_true = y_true, y_pred = y_pred, classes = ['cats', 'dogs'], normalize = 'true')

def main():
    test_robustness(path_list = ['cat2dog/testA', 'cat2dog/testB'])

if __name__ == '__main__':
    main()
