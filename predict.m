function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% e=1;
% 
% for e=1:m
%     p(e) = sigmoid(X(e)*theta)
%     if p(e) >= .5
%         p(e) = 1;
%     else
%         p(e) = 0;
%     end
% end


p=X*theta>0 %don't need sigmoid function, because any positive values will
%greater than .5 and any negative values will be less than.  If the
%statement is true, it will return a 1, other wise a 0.  It will return a
%100x1 matrix

% =========================================================================


end
