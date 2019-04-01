# B.Tech Final Year Project - Handwritten Digits Recognition Using Neural Networks
# Duration: Oct 2016 – Jun 2017

The purpose of this project is to take handwritten digits as input, process the digits, train the neural network algorithm with the processed data, to recognize the pattern and successfully identify the test digits. The popular MNIST dataset is used for the training and testing purposes. The IDE used is MATLAB

# Procedure in short
This project presents a study on recognition of handwritten digits (from 0 to 9) using neural networks from the popular MNIST dataset. I built a two-layer neural network (one hidden layer) and tried different techniques and observed the training set accuracy and test set accuracy. The final network of connections and weights obtained by using 200 hidden neurons and training for 1000 iterations using the 'fmincg' optimization algorithm and regularisation parameter = 0.1, I achieved 98.29% test set accuracy, ie. Test Set Error Percent = 1.71, which is on par with the accuracies achieved using similar classifiers by LeCun and Simard. Other than the 'fmincg' function, I have not used any prebuilt functions. My model: 2-layer NN, 200 HU, Cross-Entropy Loss (without any preprocessing).

# File Info
#### Start with the neuralNetwork.m code in _Everything Else_ folder. This neuralNetwork.m uses many functions, many of whcih are defined in the other .m files in the  _Everything Else_ folder. Adequate comments are provided in the right places for understanding the code.  
  
* neuralNetwork.m -> Main code  
  
* .idx3-ubyte -> Original training images and labels  
  
* loadMNISTImages.m  
loadMNISTLabels.m -> Load the training images and their labels from the original idx3-ubyte files  
  
* displayData.m -> Display 2D data in a nice grid  
  
* nnCostFunction.m -> Implements the neural network cost function for a two layer neural network which performs classification  
  
* sigmoidGradient.m -> returns the gradient of the sigmoid function  
  
* sigmoid.m -> Compute sigmoid function  
  
* randInitializeWeights.m -> Randomly initialize the weights of a layer of neurons in the neural network  
  
* checkNNGradients.m -> Creates a small neural network to check the backpropagation gradients  
  
* validationCurveHoldout.m -> Generate the training and validation errors needed to plot a validation curve that we can use to select regularization parameter  
  
* fmincg.m -> a function which works similarly to "fminunc"  
  
* predict.m -> Predict the label of an input given a trained neural network  
  
* predictExternalImage.m -> Predicts what digit is contained in an external image given a trained neural network  
  
* predictPercentExternalImage.m -> Predicts what digit is contained in an external image, along with the probability given a trained neural network  
  
* neuralNetworkWeights - HU200_EP1000_R0.1.mat  
neuralNetworkWeights-98.35.mat  
savedNeuralNetworkWeights.mat -> Saved Neural Network weightsz
