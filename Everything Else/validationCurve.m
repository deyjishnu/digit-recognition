function [lambda_vec, error_train, error_val] = ...
    validationCurve(input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, MaxIter, num_folds)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(input_layer_size, hidden_layer_size, ...
%       num_labels, X, y, MaxIter, num_folds) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda.
%

% Number of images in X
m = size(X, 1);

% Number of images in one fold
one_fold = m/num_folds;

% Selected values of lambda (we should not change this)
lambda_vec = [0.1 0.3 1 3]';

% We need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== CODE ======================
% Instructions: This function will return training errors in 
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
    for j = 1:num_folds
        a = (j - 1) * one_fold + 1;
        b = j * one_fold;
        Xval = X(a:b, :);
        yval = y(a:b, :);
        Xtrain = [X(1:(a-1), :); X((b+1):m, :)];
        ytrain = [y(1:(a-1), :); y((b+1):m, :)];
        [nn_params] = trainNeuralNetwork(input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, MaxIter, lambda);
                               
        error_train(i) = error_train(i) + nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   Xtrain, ytrain, 0);
                               
        error_val(i)= error_val(i) + nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   Xval, yval, 0);
    end
    
    error_train = error_train/num_folds;
    error_val = error_val/num_folds;
    
end
% =========================================================================

end
