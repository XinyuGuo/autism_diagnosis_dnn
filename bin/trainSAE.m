function [saeOptTheta] = trainSAE(paras,train)
	addpath minFunc/
	[cost,grad]	= sparseAutoencoderCost(paras.saeTheta,paras.inputSize,paras.hiddenSize,paras.lambda,paras.sparsityParam,paras.beta,train);
			
	[saeOptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p,paras.inputSize,paras.hiddenSize,paras.lambda,paras.sparsityParam,paras.beta,train),paras.saeTheta,paras.options); 
end
