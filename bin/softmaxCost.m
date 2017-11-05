function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);% Debug :100 samples

% - debug 
 %disp (size(labels));
 %disp(numCases);
 %disp(numClasses);
 %disp(size(data));
% pause;
groundTruth = full(sparse(labels,1:numCases,1));
 %disp(size(groundTruth));

cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
exponent = theta*data; %10*100
exponent = bsxfun(@minus,exponent,max(exponent,[],1));
denominator = repmat(sum(exp(exponent)),numClasses,1);
numerator = exp(exponent);
p = numerator./denominator;

%disp(size(p));

weight_decay = (lambda/2.)*sum(sum(theta.^2));
cost = -(1./numCases)*sum(sum(groundTruth.*log(p)))+weight_decay;
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
thetagrad = -(1./numCases)*(groundTruth-p)*data'+lambda*theta;
grad = [thetagrad(:)];
end
