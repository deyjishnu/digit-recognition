%% ============= Create the MNISTDataset.mat from ubyte files =============
% Requires the files:
% train-images.idx3-ubyte
% train-labels.idx1-ubyte
% t10k-images.idx3-ubyte
% t10k-labels.idx1-ubyte
% You can download these files from http://yann.lecun.com/exdb/mnist/

%% Initialization
clear ; close all; clc

fprintf('Loading and Saving Data ...\n')

% Load the training images and their labels
trainingImages = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');

% Change the labels which are '0' to '10'
trainingLabels(trainingLabels == 0) = 10;

% Load the test images and their labels
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Change the labels which are '0' to '10'
testLabels(testLabels == 0) = 10;

% Save the train images and labels, and test images and labels to file:
% 'MNISTDataset'
save('MNISTDataset');

fprintf('Done\n')