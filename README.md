The purpose of this project is to take handwritten digits as input, process the digits, train the neural network algorithm with the processed data, to recognize the pattern and successfully identify the test digits. The popular MNIST dataset is used for the training and testing purposes. The IDE used is MATLAB

# B.Tech Final Year Project - Handwritten Digits Recognition Using Neural Networks
# Duration: Oct 2016 – Jun 2017

# Procedure in short
This project presents a study on recognition of handwritten digits (from 0 to 9) using neural networks from the popular MNIST dataset. I built a two-layer neural network (one hidden layer) and tried different techniques and observed the training set accuracy and test set accuracy. The final network of connections and weights obtained by using 200 hidden neurons and training for 1000 iterations using the 'fmincg' optimization algorithm and regularisation parameter = 0.1, I achieved 98.29% test set accuracy, ie. Test Set Error Percent = 1.71, which is on par with the accuracies achieved using similar classifiers by LeCun and Simard. Other than the 'fmincg' function, I have not used any prebuilt functions. My model: 2-layer NN, 200 HU, Cross-Entropy Loss (without any preprocessing).
