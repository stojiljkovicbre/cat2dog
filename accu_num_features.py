# from utils import set_params
from src.GDA import LDA, test_LDA, set_params
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Wavelets:

        Haar ('haar')
        Daubechies ('db')
        Symlets ('sym')
        Coiflets ('coif')
        Biorthogonal ('bior')
        Reverse biorthogonal ('rbio')
        “Discrete” FIR approximation of Meyer wavelet ('dmey')
        Gaussian wavelets ('gaus')
        Mexican hat wavelet ('mexh')
        Morlet wavelet ('morl')
        Complex Gaussian wavelets ('cgau')
        Shannon wavelets ('shan')
        Frequency B-Spline wavelets ('fbsp')"""

    wavelets = ['haar',
                'db',
                'sym',
                'coif',
                'bior',
                'rbio',
                'dmey',
                'gaus',
                'mexh',
                'morl',
                'cgau',
                'shan',
                'fbsp']
    num_features = [1, 2, 5, 10, 20, 50, 100]
    num_features = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    data = []
    for wave in wavelets[0:1]:
        set_params(m = np.array(None), wave = wave)
        accu_train = []
        accu_test = []
        for n in num_features:
            accu = LDA(['cat2dog/trainA', 'cat2dog/trainB'], num_features = n, show = False)
            accu_train.append(accu)
            accu = test_LDA(path_list = ['cat2dog/testA', 'cat2dog/testB'], show = False)
            accu_test.append(accu)
        d = {'wavelet': wave, 'num_features': num_features, 'accu_train': accu_train, 'accu_test': accu_test}
        data.append(d)

    with open('train_test_data.json', 'w') as f:
        json.dump(data, f)
    print('data saved')

    # fig, ax = plt.subplots()
    # for d in data:
    #     ax.plot(d['num_features'], d['accu_train'], label = 'train')
    #     ax.plot(d['num_features'], d['accu_test'], label =  'test')
    # legend = ax.legend(loc = 'upper right', shadow = True, fontsize = 'x-large')
    # plt.show()
    print('KRAJ')

if __name__ == '__main__':
    main()
