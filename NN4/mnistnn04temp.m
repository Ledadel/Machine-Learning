% This is a 2 layer neural net with a single hidden layer to classify
% digits.  It works with the mnist training data.  It is a little slow and
% so could be improved, though this could be due partially to the
% additional pixels.

%   ( ^_^ )

%% Initialization
clear ; close all; clc  % close everything else up

%% Define the size of the Network
input_layer_size  = 784;  % 28x28 Input Images of Digits from MNIST
hidden_layer_size = 25;   % # hidden units [also changes loading params]
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% ================ Loading Training Data ==============

% Attempting to use the real MNIST data from Yann Lecun's site.  The
% following code is from UFLDL site:
X = loadMNISTImages('train-images.idx3-ubyte');  
y = loadMNISTLabels('train-labels.idx1-ubyte');


X=reshape(X,784,[]); %this reshapes the 28x28x[examples] value matrix from mnist
% and converts it into a 784x[number of examples] matrix, sort of unrolled
X=X'; % now each row is an example rather than each column.

m = size(X,1); % size of training number of examples

TrainingSet = 5000; %training set size, number of examples to train with
TestSet = 2000; %Test set size, for seeing if over or under fitting at end
ts=m-TestSet; % Test Set START position in matrix

R=randperm(size(X,1)); % create a randomization vector for randomizing the
% order of the training data and the labels
%R=[1:size(X)];  % use this to make R non-random for swapping during
%testing

% ======  X  =======
Xr=X(R,:);  % X random will randomize the order of the training data
X=Xr(1:TrainingSet,:);  % using only a subset of the training examples
Xt=Xr(ts:m,:);  % Xt = X test set


% ======  Y  =======
% Adjusting labels of "0" into "10" because this is the index of the
% ymatrix later on in nnCostfunction.
for i=1:length(y)
    if y(i) == 0;
       y(i)=10;
    end;    
i=i+1;
end


yr=y(R,:); % randomizing the y labels in the same way as the X training
y=yr(1:TrainingSet,:);  % changing the number of training examples
yt=yr(ts:m,:); %y label TEST set, still randomized.


% ===== Display Some Examples  ========
%X=X(1:TrainingSet,:);  %this should change it the the same size as original nn01
% grabbing the first 5000 training examples
   sel = randperm(size(X, 1),784);  %randperm selects k from 1-n interger
    displayData(X(sel, :));
%    fprintf('Program paused. Press enter to continue.\n');
%    pause;


size(X) % checking the size of the training set.

%y=y(1:TrainingSet,:);  %cropping size of y labels so we don't have to run all of them
ymat=zeros(length(y),num_labels); %ymatrix is inialized as a zeros matrix
for i=1:length(y)
    indx=y(i);
    ymat(i,indx)=1;
i=i+1;
end

ytmat=zeros(length(yt),num_labels); %ymatrix is inialized as a zeros matrix
for i=1:length(yt)
    indx=yt(i);
    ymat(i,indx)=1;
i=i+1;
end

%size(y)  % checking size of labels.  this is 60,000x1
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
lambda = 5;  %regularization parameter

tic;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, ymat, lambda...
                                   Xt, ytmat);
                               
% %calculate cost of test set:
% testCostFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
%                 num_labels, Xt, ytmat, lambda);
                    
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
plot(costx,cost, '.', costx,costT,'+'); % attempting to find cost of test set each time
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


pred = predict(Theta1, Theta2, X); % for getting prediction accuracy the old way
predt = predict(Theta1, Theta2, Xt); % using real test set to get prediction accuracy
fprintf('\n(Training Set Accuracy: %f)\n', mean(double(pred == y)) * 100);
fprintf('\nTEST Set Accuracy: %f\n', mean(double(predt == yt)) * 100);

%=======
