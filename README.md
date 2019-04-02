# B.Tech Final Year Project - Handwritten Digits Recognition Using Neural Networks
# Duration: Oct 2016 – Jun 2017

The purpose of this project is to take handwritten digits as input, process the digits, train the neural network algorithm with the processed data, to recognize the pattern and successfully identify the test digits. The popular MNIST dataset is used for the training and testing purposes. The IDE used is MATLAB

# Procedure in short
This project presents a study on recognition of handwritten digits (from 0 to 9) using neural networks from the popular MNIST dataset. I built a two-layer neural network (one hidden layer) and tried different techniques and observed the training set accuracy and test set accuracy. The final network of connections and weights obtained by using 200 hidden neurons and training for 1000 iterations using the 'fmincg' optimization algorithm and regularisation parameter = 0.1, I achieved 98.29% test set accuracy, ie. Test Set Error Percent = 1.71, which is on par with the accuracies achieved using similar classifiers by LeCun and Simard. Other than the 'fmincg' function, I have not used any prebuilt functions. My model: 2-layer NN, 200 HU, Cross-Entropy Loss (without any preprocessing).

# Test Set Accuracy achieved with various parameters on the MNIST dataset


Number of hidden neurons | Number of iterations (epochs) | Training Set Accuracy (%) | Test Set Accuracy (%)
---------- | ---------- | ---------- | ----------
25 | 30 | 89.9533 | 90.18
50 | 30 | 91.54 | 91.57
100 | 30 | 90.29 | 90.94
200 | 100 | 96.261667 | 96.07
200 | 1000 | 100 | __98.29__
---
We achieved maximum accuracy of 98.29% (error = 1.71%) on the test set of the MNIST dataset without using any ready-made toolkit. 

# Code Info (or How to run the code)
#### Start with the [_neuralNetwork.m_](https://github.com/deyjishnu/digit-recognition/blob/master/Code/neuralNetwork.m) code in the [_Code_](https://github.com/deyjishnu/digit-recognition/blob/master/Code) folder. This neuralNetwork.m uses many functions, many of which are defined in the other .m files in the _Code_ folder. Adequate comments are provided in the right places for understanding the code. 
  
* [_neuralNetwork.m_](https://github.com/deyjishnu/digit-recognition/blob/master/Code/neuralNetwork.m) -> Main code  
  
* _.idx3-ubyte_ -> Contains the original training images and labels  
  
* _[loadMNISTImages.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/loadMNISTImages.m), [loadMNISTLabels.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/loadMNISTLabels.m)_ -> Loads the training images and their labels from the original idx3-ubyte files  
  
* _[displayData.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/displayData.m)_ -> Displays 2D data in a nice grid  
  
* _[nnCostFunction.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/nnCostFunction.m)_ -> Implements the neural network cost function for a two layer neural network which performs classification  
  
* _[sigmoidGradient.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/sigmoidGradient.m)_ -> Returns the gradient of the sigmoid function  
  
* _[sigmoid.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/sigmoid.m)_ -> Computes sigmoid function  
  
* _[randInitializeWeights.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/randInitializeWeights.m)_ -> Randomly initialize the weights of a layer of neurons in the neural network  
  
* _[checkNNGradients.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/checkNNGradients.m)_ -> Creates a small neural network to check the backpropagation gradients  
  
* _[validationCurveHoldout.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/validationCurveHoldout.m)_ -> Generates the training and validation errors needed to plot a validation curve that we can use to select regularization parameter  
  
* _[fmincg.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/fmincg.m)_ -> a function which works similarly to "fminunc"  
  
* _[predict.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/predict.m)_ -> Predicts the label of an input given a trained neural network  
  
* _[predictExternalImage.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/predictExternalImage.m)_ -> Predicts what digit is contained in an external image for a trained neural network  
  
* _[predictPercentExternalImage.m](https://github.com/deyjishnu/digit-recognition/blob/master/Code/predictPercentExternalImage.m)_ -> Predicts what digit is contained in an external image, along with the probability given a trained neural network  
  
* _neuralNetworkWeights - HU200_EP1000_R0.1.mat, neuralNetworkWeights-98.35.mat, savedNeuralNetworkWeights.mat_ -> Saved Neural Network weights

# Sample Images
Image sets present inside the [_Sample Images_](https://github.com/deyjishnu/digit-recognition/tree/master/Sample%20Images) folder are created using Microsoft Paint. These are 28x28 pixel images of digits 0-9 to mimic the MNIST dataset. These images are used for external image recognition. 

# Where to learn the Basics and More
We have a detailed guide for the project in [_A layman's guide to the project_](https://github.com/deyjishnu/digit-recognition/blob/master/A%20layman's%20guide%20to%20the%20project.pdf). 
Two research papers which helped us:
1. [Handwritten Digit Recognition with a Back-Propagation Network](https://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network) by LeCun et al
2. [A Set of Features Extraction Methods for the Recognition of the Isolated Handwritten Digits](https://www.researchgate.net/profile/Salim_Ouchtati/publication/271908379_A_Set_of_Features_Extraction_Methods_for_the_Recognition_of_the_Isolated_Handwritten_Digits/links/57ec4a4d08aebb1961ffa8f5/A-Set-of-Features-Extraction-Methods-for-the-Recognition-of-the-Isolated-Handwritten-Digits.pdf) by _Ouchtati, S and Redjimi, M and Bedda, M_

If you want to learn what neural networks are and how to start with Matlab, you can start with this popular [course](https://www.coursera.org/learn/machine-learning) by Andrew Ng.

# Acknowledgement 

Effective noise removal from external images is a very complex process and
requires in-depth knowledge in this domain. And without good noise
removal, it is impossible to achieve good success rate in detecting
digits from external images.

As we didn’t prioritise on processing of external images over getting good results on test data set, the results on external images is not good and is inconsistent.

The segmentation algorithm can also be vastly improved to identify individual
objects from all types of images, without false positives.
