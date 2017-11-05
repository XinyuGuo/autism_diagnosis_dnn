% % % % % % % % % %
% Project: Autism
% Author: Xinyu Guo
% Date: 6-9-2016
% train one sae and get its opttheta containing weights and bias
% usage : getOriFeatureSets(saeparameters,opttheta_parameters,training_data)
% % % % % % % % % % 

function [opttheta]= getFeaturesFromSAE(saeParas,optParas,trainingset)
	%parameters setting
	visibleSize = saeParas{1,1};   
	hiddenSize = saeParas{2,1};     
	sparsityParam = saeParas{3,1}; 
	lambda = saeParas{4,1};     
	beta = saeParas{5,1};           
	%initializing weights
	theta = initializeParameters(hiddenSize, visibleSize);

	disp('theta')
	disp(size(theta))
	disp('traiingset')
	disp(size(trainingset))	

	[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda,sparsityParam, beta, trainingset');

	%setting parameters for the optimization function
	addpath minFunc/
	options.Method = optParas{1,1};
	options.maxIter = optParas{2,1}; 
	options.display = optParas{3,1};
	% learning 
	[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                               visibleSize, hiddenSize, ...
                               lambda, sparsityParam, ...
                               beta, trainingset'), ...
                               theta, options);
end
