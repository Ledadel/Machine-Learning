function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections for the neural net.
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W is set to a matrix of size(L_out, 1 + L_in) as
%   the column of W handles the "bias" terms

%The first column of W is a ones column that corresponds to the 
%parameters for the bias units



% newer random matrix creation, removing epsilon
W = .2*randn(L_out, 1+L_in);


% =========================================================================

end
