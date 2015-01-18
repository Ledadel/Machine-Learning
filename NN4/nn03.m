% This is a 2 layer neural net with a single hidden layer to classify digits.
% This actually splits the ex4data.mat into a training set and a test set
%   ( ^_^ )

%% Initialization
clear ; close all; clc  % close everything else up

%% Define the size of the Network
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % # hidden units [also changes loading params]
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% ================ Loading Training Data ==============

load('ex4data1.mat');  % training digits in 20x20pixel images
R=randperm(size(X,1)); % create a randomization vector for randomizing the
% order of the training data and the labels
%R=[1:size(X)];  % use this to make R non-random for swapping during
%testing

Xr=X(R,:);  % X random will randomize the order of the training data
X=Xr(1:4000,:);  % using only a subset of the training examples
Xt=Xr(4001:5000,:);  % Xt = X test set
yr=y(R,:); % randomizing the y labels in the same way as the X training
y=yr(1:4000,:);  % changing the number of training examples
yt=yr(4001:5000,:); %y label TEST set, still randomized.

%size(X)
%size(y)
m = size(X,1); % size of ex4data.mat number of examples
%pause;

%% ================ Initializing Pameters ================
% The function 'randInitializeWeights.m' creates the initial weights of the 
% neural network including the bias terms.  These are later unrolled into a
% vector for arguments of functions and plotting.


% Initializing Neural Network Parameters
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);

% Unroll parameters into vector
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params=initial_nn_params;  % future function calls use nn_params, so I'm
% creating this from randomized weights of arbitrary size.

% m: this is to see what the initial weights look like.
figure
histogram(nn_params')
hold on;

%% =================== Training NN ===================
%  These advanced optimizers are able to train our cost functions
%  efficiently as long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 100); %do not understand how this works, but 
%fmincg somehow
%  Test different values of lambda
lambda = 10;  %regularization parameter

tic;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost,i1,xh] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ================= Visualize Weights and Cost Function =================

% m: The following will produces a graph of the cost vs iteration
cost=cost'; % make vector of cost history vs iteration
costx=(1:i1)'; % create a vector of x values the number of iterations long
figure;
plot(costx,cost, '.')
ylabel('Cost');
xlabel('Iterations');
hold on;

% m: The following will produce a graph of the history of the weights from
% the matrix xh (X History)
%figure;
%plot(xh')  % m: running this will add 4-5seconds to training.
%hold on;
toc

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% m: this would be nice to have a test set to see what our actual training
% accuracy was.

pred = predict(Theta1, Theta2, X); % for getting prediction accuracy the old way
predt = predict(Theta1, Theta2, Xt); % using real test set to get prediction accuracy
fprintf('\n(Training Set Accuracy: %f)\n', mean(double(pred == y)) * 100);
fprintf('\nTEST Set Accuracy: %f\n', mean(double(predt == yt)) * 100);

%=======
