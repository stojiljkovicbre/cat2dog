# cat2dog
cat2dog using DWT, SVD and GDA

This is an algorithm that classifies images of cats and dogs.
Algorithm uses 2D Discrete Wavet Transform to extract features, Singular Value Decomposition to reduce dimensionality
and Gaussan Discriminant Analysis (or Logistic Regression or Support Vector Machine) to classify image.

This project uses cat2dog dataset from Kaggle: https://www.kaggle.com/waifuai/cat2dog.

First of all install all necessary software:
	- install anaconda from official website https://www.anaconda.com/ ans add it to your PATH
	- execute SETUP_CONDA_ENVIRONMENT.bat and make sure it endas with: 'cat2dog environment has been set up'

To train GDA algorithm:
	- execute train_GDA.bata

To evaluate performances of trained GDA algorithm
	- execute test_GDA.bata

To test the algorithm on a single image use cat2dog_app.bat:
	- select and load image
	- select classifier (LDA(GDA)/SVM/LogReg)
	- mark animals right eye
	- mark animals left eye
	- hit classify
