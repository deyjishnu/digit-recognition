function [percent, predict] = predictPercentage(Theta1, Theta2, X)
%PREDICTPERCENTAGE Predict the label of an input and the probability given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X and its probability given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
predict = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[percent, predict] = max(h2, [], 2);
end