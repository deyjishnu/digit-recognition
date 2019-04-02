function [nn_params, J_history] = gradientDescent(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, alpha, MaxIter)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   [nn_params, J_history] = gradientDescent(nn_params, ...
%   input_layer_size, ...
%   hidden_layer_size, ...
%   num_labels, ...
%   X, y, lambda, alpha, MaxIter) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
J_history = zeros(MaxIter, 1);

for iter = 1:MaxIter

    [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda);
    % ====================== CODE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    nn_params = nn_params - alpha*grad;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = J;
    
    fprintf('Iteration \t | Cost: %4.6e\n', J);
end

end
