function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h_theta = sigmoid(X * theta);
cost = -y .* log(h_theta) - (1 - y) .* log(1 - h_theta);
J = 1 ./ m * sum(cost);
JRegTerm = lambda / (2 * m) * sum(theta(2 : end) .^ 2);
J = J + JRegTerm;

sizeX = size(X);
diff = repmat((h_theta - y), 1, sizeX(2));
grad = (1 ./ m .* sum( diff .* X))';
gradZero = grad(1);
grad = grad + lambda / m .* theta; % Adding Regularzation
grad(1) = gradZero; % recover grad(0) since it shouldn't adding reg for theta(0)

% =============================================================

end
