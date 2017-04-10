function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%printf("num_iter %d\n", num_iters);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp = zeros(size(theta, 1), size(theta, 2))
    for idx = 1:size(theta)
      temp(idx) = theta(idx) - alpha*(1/m)*sum( (X*theta - y).*X(:, idx) );
    end
    
    for idx = 1:size(theta) 
      theta(idx) = temp(idx)
    end

    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %printf("J = %d\n", J_history(iter));

end

end
