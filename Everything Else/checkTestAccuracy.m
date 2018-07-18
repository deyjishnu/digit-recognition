%% ================= Implement Predict for Test Set =================
%  Now we would like to use it to predict the test(or validation) labels.
%  You will now implement the "predict" function to use the neural network to
%  predict the labels of the test set. This lets you test the neural network
%  on images on which it is not trained.

% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.

% Load Test Data
fprintf('Loading and Visualizing Data ...\n')

load('MNISTDataset');
testX = testImages((1:10000),:);
testy = testLabels((1:10000),:);
n = size(testX, 1);
load('neuralNetworkWeights.mat');

% Randomly select 100 data points to display
sel = randperm(size(testX, 1));
sel = sel(1:100);

displayData(testX(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, testX);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == testy)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Predicting test images =============
%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(n);

for i = 1:n
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(testX(rp(i), :));

    pred = predict(Theta1, Theta2, testX(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end