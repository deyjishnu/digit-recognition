function [nn_params, cost] = trainNeuralNetwork(input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, MaxIter, reg_param)
%TRAINLINEARREG Trains neural network given a dataset (X, y) and a
%regularization parameter lambda
%   [nn_params, cost] = trainNeuralNetwork(input_layer_size,
%   hidden_layer_size, num_labels, X, y, MaxIter, reg_param)
%   trains the neural network using the dataset (X, y) and
%   regularization parameter lambda. Returns the
%   trained parameters theta.
%

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

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

end
