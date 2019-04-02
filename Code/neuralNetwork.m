function neuralNetwork(hidden_layer_size, num_train_images, num_test_images, MaxIter, reg_param)
% LOADDATA Loads the dataset of MNIST
%   function neuralNetwork(hidden_layer_size, num_train_images, num_test_images, MaxIter, reg_param)
%   is used to run all the parts of our neural network.

%% Machine Learning - Neural Network Learning & Prediction

%  Instructions
%  ------------
% 
%  This file can run all the codes and programs of the neural network
%  without user intervention. After running each part of this code, the 
%  function pauses and waits for the user to continue with the next part.
%  Currently this function requires the following created functions:

%       displayData.m -> Display 2D data in a nice grid
%       nnCostFunction.m -> Implements the neural network cost function for a two layer neural network which performs classification
%       sigmoidGradient.m -> returns the gradient of the sigmoid function
%       randInitializeWeights.m -> Randomly initialize the weights of a layer of neurons in the neural network
%       checkNNGradients.m -> Creates a small neural network to check the backpropagation gradients
%       validationCurveHoldout.m -> Generate the training and validation errors needed to plot a validation curve that we can use to select regularization parameter
%       fmincg.m -> a function which works similarly to "fminunc"
%       predict.m -> Predict the label of an input for a trained neural network

%% Initialization
clear ; close all; clc

if ~exist('hidden_layer_size', 'var') || isempty(hidden_layer_size)
    hidden_layer_size = 25;
end

if ~exist('num_train_images', 'var') || isempty(num_train_images)
    num_train_images = 60000;
end

if ~exist('num_test_images', 'var') || isempty(num_test_images)
    num_test_images = 10000;
end

if ~exist('MaxIter', 'var') || isempty(MaxIter)
    MaxIter = 30;
end

if ~exist('reg_param', 'var') || isempty(reg_param)
    reg_param = 0.1;
end

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by first loading and visualizing the dataset. 
%  We will be working with a dataset that contains handwritten digits.

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('MNISTDataset');
X = trainingImages((1:num_train_images),:);
y = trainingLabels((1:num_train_images),:);
testX = testImages((1:num_test_images),:);
testy = testLabels((1:num_test_images),:);

m = size(X, 1);
n = size(testX, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Parameters ================
% In this part, we load some pre-initialized 
% neural network parameters.

% Initialize variables Theta1 and Theta2
Theta1 = zeros(hidden_layer_size, input_layer_size + 1);
Theta2 = zeros(num_labels, hidden_layer_size + 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('savedNeuralNetworkWeights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, we should first start by implementing the
%  feedforward part of the neural network that returns the cost only. We
%  should run the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, we can verify that
%  our implementation is correct by verifying that we get the same cost
%  as for the fixed debugging parameters.
%
%  We will implement the feedforward cost *without* regularization
%  first so that it will be easier for us to debug. Later, in part 4, we
%  will implement the regularized cost.
%
fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from savedNeuralNetworkWeights): %f '...
         '\n(this value should be about 0.153726)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Implement Regularization ===============
%  Once our cost function implementation is correct, we should now
%  continue to implement the regularization with the cost.
%

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from savedNeuralNetworkWeights): %f '...
         '\n(this value should be about 0.192581)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 5: Sigmoid Gradient  ================
%  Before we start implementing the neural network, we will first
%  implement the gradient for the sigmoid function.

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Initializing Parameters ================
%  In this part, we will be starting to implement a two
%  layer neural network that classifies digits. We will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Part 7: Implement Backpropagation ===============
%  Once our cost calculation is complete, we should implement the
%  backpropagation algorithm for the neural network. We should add to the
%  code we've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 8: Implement Regularization ===============
%  Once our backpropagation implementation is correct, we should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.270292)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 9: Validation for Selecting Lambda =============
%  We will now implement validationCurve to test various values of 
%  lambda on a validation set. We will then use this to select the
%  "best" lambda value.
%

% K-fold Cross Validation
% num_folds = 6;
% [lambda_vec, error_train, error_val] = validationCurve(input_layer_size, hidden_layer_size, ...
%                                   num_labels, X, y, MaxIter, num_folds);

[lambda_vec, error_train, error_val] = validationCurveHoldout(input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, MaxIter, 0.3);


close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 10: Training NN ===================
%  We have now implemented all the code necessary to train a neural 
%  network. To train the neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". These
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  Afterwards change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', MaxIter);

%  We should also try different values of lambda
lambda = reg_param;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save('neuralNetworkWeights.mat','Theta1','Theta2')
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 11: Visualize Weights =================
%  We can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

% Switch on the below line afterwards
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 12: Implement Predict for Training Set =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  us compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give an idea of the network's output, we can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 13: Implement Predict for Test Set =================
%  Now we would like to use it to predict the test(or validation) labels.
%  We will now implement the "predict" function to use the neural network to
%  predict the labels of the test set. This lets us test the neural network
%  on images on which it is not trained.

pred = predict(Theta1, Theta2, testX);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == testy)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To giv an idea of the network's output,we can also run
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

end