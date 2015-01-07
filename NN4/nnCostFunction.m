function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables:
[m,~] = size(X);  %m is the number of training examples
         


% Later this functions implements the backpropagation algorithm to compute 
% the gradients Theta1_grad and Theta2_grad. This returns the partial derivatives 
% of the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% Theta2_grad, respectively. 
%
%

%--------------------------------------------------------
%Part 1:  Forward Propagation of the Neural Network:
a0=X;  %5000x400, a0 = X is the training data and input to the network
a0b=[ones(m,1) a0]; %5000x401, =a0+bias values, a0= 5000x400; ones column added 
%for the bias values each neuron which has 401 inputs per example (400 pixels, 1 bias)
a0b=a0b'; %401x5000
z1=Theta1*a0b;  %z1 = 25x5000, theta1 = 25x401,  a0 = 401x5000 /1st step in forward prop
a1=sigmoid(z1); %a2 = 25x5000; nonlinear activation on z input
a1b=[ones(1,m); a1];  %26x5000; a1b = a1+bias rows
z2=Theta2*a1b; % z2 = 10x5000, Theta2 = 10 x 26,  a2 = 26x5000      
a2=sigmoid(z2); % 10x5000 final output
h=a2'; % h = 5000x10; h is also the hypothesis of the NN


%----------------------------------------------------------
%Part 2: Calculates cost(J) based and the prediction (h) and the actual label 
% data (y).  The following first converts the label data into a binary label
%matrix which then the cross entropy function is used for the cost.


% Creates ymatrix of labels from y that has a 1 in the label column and a
% zero everywhere else.  So ymatrix is 5000 rows, 10 columns with a 1 in the 
% correct label column which contains the same information as in the y
% label vector.

ymatrix=zeros(length(y),num_labels); %ymatrix is inialized as a zeros matrix

%idx is an index vector that specificies the locations to place a 1 based
%on the y vector
idx=sub2ind(size(ymatrix),1:length(y),y'); %sub2ind function does this indexing

ymatrix(idx)=1; %converts the indexed spots to 1 labels


%Calculates cost using cross entropy cost function
%Jmatrix is element wise multiplication giving cost per prediction for all examples
Jmatrix=-(ymatrix.*log(h)+(1-ymatrix).*log(1-h)); %Jmatrix (5000x10)
                                                                                      
Jvector=Jmatrix*ones(num_labels,1); %sums up all the costs per example into a single vector
                                                
J=1/m*ones(1,length(y))*Jvector; %This is the average cost per example without regularization

%-----------------------------------------------------------
%Part 3: Calculating the cost with regularization.  This just calculates
%the cost function when using L2 regularization, but this part is not used
%for updating any weights as the derivative can be taken explicitely and
%the value of the weight added to the gradient as is done in section 5.

% Regularization: adding regularization - - - - - - - - - - - - - - - - - -
Thet1wob=Theta1(:,2:end);  % creating variables without bias (first 1's column).
Thet2wob=Theta2(:,2:end);  % creating variables without bias (first 1's column).


t1sqr=Thet1wob.^2; %element wise squaring the weights without the bias
t2sqr=Thet2wob.^2; %element wise squaring the weights without the bias

%The below sums all the squared values fo the weights using
%vector-matrix-vector multiplication
sumt1sqr=ones(1,size(t1sqr,1))*t1sqr*ones(size(t1sqr,2),1); %vector-matrix-vector
sumt2sqr=ones(1,size(t2sqr,1))*t2sqr*ones(size(t2sqr,2),1); %vector-matrix-vector

reg=lambda/(2*m)*(sumt1sqr+sumt2sqr); %adding and averaging

J = J + reg; % final cost with regularization

% -------------------------------------------------------------
%%=====Part 4: Backpropagation Algorithm 
% This find the gradients of the cost function with 
%respect to the weights:  Theta2_grad and Theta1_grad
%In the comments, w is used for the weights or theta, C is for the cost function
%The cross entropy cost function is used which makes dC/dz2 a simpler
%expression that is simplified analytically.

%Finding Theta2_grad
d2=h-ymatrix; %5000*10 =dC/dz2; for every example of last 
%layer neurons.  The gradeint of sigmoid(z) is already included in this.
Theta2_grad=(a1b*d2)'/m; %26*10= dC/dw2; a1b= dz2/dw2; d2=dC/dz2


%Finding Theta1_grad
dCda1=d2*Thet2wob; %5000x25 = dC/da1; d2 =dC/dz2; dz2/da1=t2wob or w2 matrix; 
d1=dCda1.*sigmoidGradient(z1)'; %5000*25 = dC/dz1,  delement wise product for all the examples
                                % dCda1=dC/da1; da1/dz1=sigmoidGradient(z1)

Theta1_grad=(a0b*d1)'/m; %401*25 dC/dw1 [matrix of the partial derivitives]
                          % a0b=dz1/dw1; d1=dC/dz1

%-----------------------------------------------------------
%Part 5: Regularization:  The following implements L2 regularization on the
%  gradients.  It is calculated by taking the derivative of the cost
%  function analytically that adding the weight to the gradient implements L2
%  regularization.  Reg1 and Reg2 are proportional to the value of the
%  current weights.


%  regularized portions
reg1 = (lambda/m)*(Theta1(:,2:end)); %Theta1/wob *regularization paramater
reg2 = (lambda/m)*(Theta2(:,2:end)); %Theta2/wob *regularization parameter
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + reg1;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + reg2;




% =========================================================================

% Put together grad: this is the partial derivatives and is the main output
% of the function.
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
