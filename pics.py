from src.utils import (center_image_eyes, set_params, get_params, basic_preprocess_image,
                       basic_preprocess_image_normal, get_features, get_features_normal,
                       load_data, do_SVD, get_accuracy, center_image_eyes, plot_conf_mat,
                       normalize_range)
from src.GDA import temp_finction
import cv2
import numpy as np
import matplotlib.pyplot as plt

S, V, DATA, labels, indices, ls, l = temp_finction()
print('V.shape', V.shape)

image = cv2.imread('maca.jpg')
image = cv2.resize(image, (500, 280))
# cv2.imshow('demo', image)
# key = cv2.waitKey()

right_eye = [111, 226]
left_eye = [134, 288]

img = center_image_eyes(image, right_eye, left_eye)
# cv2.imshow('demo', img)
# key = cv2.waitKey()

fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.reshape(V[:, i], (64, 64)), cmap = 'gray')
plt.show()

features = get_features_normal(img)
coefs = V.transpose().dot(features)

recon = []
R = np.zeros((64, 64))
for i, c in enumerate(coefs):
    R = R + c*np.reshape(V[:, i], (64, 64))
    recon.append(R)
    
fig = plt.figure()
plt.subplot(1, 6, 1)
plt.title('original')
plt.imshow(np.reshape(features, (64, 64)), cmap = 'gray')
plt.subplot(1, 6, 2)
plt.title('after 10')
plt.imshow(recon[10], cmap = 'gray')
plt.subplot(1, 6, 3)
plt.title('after 100')
plt.imshow(recon[100], cmap = 'gray')
plt.subplot(1, 6, 4)
plt.title('after 500')
plt.imshow(recon[500], cmap = 'gray')
plt.subplot(1, 6, 5)
plt.title('after 1000')
plt.imshow(recon[1000], cmap = 'gray')
plt.subplot(1, 6, 6)
plt.title('after 2000')
plt.imshow(recon[2000], cmap = 'gray')
plt.show()

O = np.zeros((512, 512, 3))
original = normalize_range(np.reshape(features, (64, 64)))
original = cv2.resize(original, (512, 512))
# out = cv2.VideoWriter('eigenfaces.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (512, 512))
for i, r in enumerate(recon):
    I = np.zeros((512, 512, 3))
    J = cv2.resize(normalize_range(r), (512, 512))
    I[:, :, 0] = J
    I[:, :, 1] = J
    I[:, :, 2] = J
    # I = np.hstack((O, I))
    cv2.imshow('Reconstucting...'.format(i+1), I)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # out.write(I)
    print('eigenfaces {}/{}'.format(i + 1, len(recon)), end = '\r')
print('\n')
# out.release()

print('KRAJ')