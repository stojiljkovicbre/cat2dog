# from utils import set_params
from src.LDA import LDA, test_LDA, set_params
import json
import numpy as np
import matplotlib.pyplot as plt

def main():

    with open('train_test_data.json') as f:
        data = json.load(f)

    d = data[0]
    fig, ax = plt.subplots()
    ax.plot(d['num_features'], d['accu_train'], label = 'train', c = '#0000ff')
    ax.plot(d['num_features'], d['accu_test'], label =  'test', c = '#ff0000')
    ax.grid()
    ax.scatter(d['num_features'], d['accu_train'], c = '#00ff00')
    ax.scatter(d['num_features'], d['accu_test'], c = '#00ff00')
    legend = ax.legend(loc = 'lower right', shadow = True, fontsize = 'x-large')
    ax.set_title('Accuracy as a function of number of fearures N')
    ax.set_xlabel('N')
    ax.set_ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    main()
