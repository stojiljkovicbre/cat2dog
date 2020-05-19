from src.GDA import LDA

def main():
    LDA(['cat2dog/trainA', 'cat2dog/trainB'], num_features = 2)

if __name__ == '__main__':
    main()
