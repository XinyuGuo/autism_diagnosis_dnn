function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data) 
 
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% Cost and gradient variables (your code needs to compute these values). 
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));%784*200
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% 1. perform a feedforward pass, computing the activations for layers L2
% L3, and so on up to the output layer Lnl.
number = size(data); % dimension : 64*10000
m = number(2); % m = 10000
%--------------------checking------------------
%m = number(2)/1000;
%B1 = zeros(hiddenSize,m);
%for i = 1:m
%   B1(:,i) = b1; 
%end
%Z2 = W1*data(:,1:10)+B1;%25 * 10000
%----------------------------------------------
%B1 = zeros(hiddenSize,m); % should be marked when checking
%disp(size(B1));
%for i = 1:m
%   B1(:,i) = b1; 
%end
%disp(size(B1));
%pause;

Z2 = W1*data+repmat(b1,1,m); % should be marked when checking
a_2 = sigmoid(Z2);
%nnz(isnan(data))
%B2 = zeros(visibleSize,m);% should be marked when checking
%for j = 1 : m
%    B2(:,j) = b2;
%end
%---------------------checking-----------------
%B2 = zeros(visibleSize,m);
%for j = 1 : m
%    B2(:,j) = b2;
%end
%----------------------------------------------
Z3 = W2*a_2+repmat(b2,1,m); % W2(64*25)
a_3 = sigmoid(Z3);
%--------------------checking------------------
%squareError = (a_3-data(:,1:10)).^2;
%----------------------------------------------
squareError = (a_3-data).^2;
singleSquareError = 0.5*sum(squareError); % 1*m 
weight_decay = sum(sum(W1.^2))+sum(sum(W2.^2));
sparsityParam_hat = mean(a_2,2); % average of activation 200*1
%disp(a_2(1,:))
kl =  sum(sparsityParam*log(sparsityParam./sparsityParam_hat)+(1-sparsityParam)*log((1-sparsityParam)./(1.-sparsityParam_hat)));
cost = (1/m)*sum(singleSquareError)+(lambda/2)*weight_decay+beta*kl;
% 2.Backpropagation
% the output layer
%--------------------checking----------------------
%error_3 = -(data(:,1:10) - a_3);
%--------------------------------------------------
error_3 = -(data- a_3); % 784*10000
delta_3 = error_3.*(a_3.*(1.-a_3));%784*10000
% the hiden layer
delta_2 = W2'*delta_3;  % 200*10000
dev = a_2.*(1-a_2);     % 200*10000
sparsityTerm = beta*((1.-sparsityParam)./(1.-sparsityParam_hat)-(sparsityParam./sparsityParam_hat)); %200*1
%sparsityMatrix = zeros(hiddenSize,m); 
%for k = 1:m
%    sparsityMatrix(:,k) = sparsityTerm;
%end
sparsityMatrix = repmat(sparsityTerm,1,m);%200*10000
delta_2 = (delta_2+sparsityMatrix).*dev; % 200*10000
% compute the desired partial derivatives 
% for iter = 1 : m
%    aa2 = a_2(:,iter); %200*1
%    aa1 = data(:,iter);%784*1
%    delta3 = delta_3(:,iter);%784*1
%    delta2 = delta_2(:,iter);%200*1
%    W2grad = W2grad+delta3*aa2';
%    b2grad = b2grad+delta3;
%    W1grad = W1grad+delta2*aa1';
%    b1grad = b1grad+delta2;  
% end

W2grad = W2grad+delta_3*a_2';
b2grad = b2grad+sum(delta_3,2);
W1grad = W1grad+delta_2*data';
b1grad = b1grad+sum(delta_2,2);
% update the parameters


W2grad = (1/m)*W2grad + lambda*W2; 
b2grad = (1/m)*b2grad; 
W1grad = (1/m)*W1grad + lambda*W1; 
b1grad = (1/m)*b1grad; 
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end
