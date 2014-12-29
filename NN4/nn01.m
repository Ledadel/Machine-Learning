% This is a neural net test where we will try to simplify it for
% understanding how it works.

%   ( ^_^ )


%% Initialization
clear ; close all; clc  % close everything else up

%% Setup the parameters (weights)
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 20;   % # hidden units [also changes loading params] 
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)


%% ================ Loading Parameters ================

load('ex4data1.mat');
m = size(X, 1);

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');



%% ================ Initializing Pameters ================
%  implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

%fprintf('\nInitializing Neural Network Parameters ...\n')


initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params=initial_nn_params;  % future function calls use nn_params, so I'm
% creating this from randomized weights of arbitrary size.

% m: this is to see what the initial weights look like.
figure
histogram(nn_params')
hold on;



%% =================== Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);
%  You should also try different values of lambda
lambda = 1;

tic;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost,i1,costy,xh] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));




%% ================= Visualize Weights and Cost Function =================


% m: The following will produces a graph of the cost vs iteration
costy=costy';
costx=(1:options.MaxIter)';  
figure;
plot(costx,costy, '.')
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


