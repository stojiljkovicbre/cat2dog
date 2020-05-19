from src.SVM import SVM

def main():
    SVM(path_list = ['cat2dog/trainA', 'cat2dog/trainB'],
        num_features = 2,
        show = True)

if __name__ == '__main__':
    main()
