function [highestfishernodes]= topNnodes(theta,modelparas,netconfig,data,labels,nodesperlayer)
	index = size(modelparas,1)-1;
	softparas = modelparas{index};
	hiddenSize = softparas.inputSize;
	numClasses = softparas.numClasses;
	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
	stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
	%size(stack{1}.w)
	%size(data)
    [B,I] = fisherscore(stack{1}.w,data',labels);
	weights = stack{1}.w;	

	highestfishernodes = weights(I(1:nodesperlayer),:);  	
end
