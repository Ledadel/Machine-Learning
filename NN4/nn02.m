% This is a 2 layer neural net with a single hidden layer to classify digits.

%   ( ^_^ )

%% Initialization
clear ; close all; clc  % close everything else up

%% Define the size of the Network
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 30;   % # hidden units [also changes loading params]
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% ================ Loading Training Data ==============

%load('ex4data1.mat');  % training digits in 20x20pixel images

% Attempting to 
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));
m = size(X, 1); % size of ex4data.mat number of examples

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

options = optimset('MaxIter', 200); %do not understand how this works, but 
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

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy (based on training set?): %f\n', mean(double(pred == y)) * 100);

%=======
