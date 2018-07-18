function [lambda_vec, error_train, error_val] = ...
    validationCurveHoldout(input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, MaxIter, validation_to_all_ratio)
%VALIDATIONCURVEHOLDOUT Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(input_layer_size, hidden_layer_size, ...
%       num_labels, X, y, MaxIter, validation_to_all_ratio) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda.
%

% Number of images in validation set
m = size(X, 1);
m_val = m * validation_to_all_ratio;

% Create the validation set and training set
Xval = X(1:m_val, :);
yval = y(1:m_val, :);
Xtrain = X(m_val + 1:end, :);
ytrain = y(m_val + 1:end, :);

% Selected values of lambda (we should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% We need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== CODE ======================
% Instructions: This function will to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               us the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: We can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training neural 
%           % network with regularization parameter lambda
%           % We should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    [nn_params] = trainNeuralNetwork(input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, MaxIter, lambda);
                               
        error_train(i) = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels, ...
                                   Xtrain, ytrain, 0);
                               
        error_val(i)= nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels, ...
                                   Xval, yval, 0);
end
% =========================================================================

end
