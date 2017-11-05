function [stackedAETheta,netconfig] = getFinetumeParas(modelParas,layerOpt,softmaxOptTheta)
	layernum = size(layerOpt,1);
	stack = cell(layernum,1);
	for i=1:layernum
		saeOpt = layerOpt{i};
		stack{i}.w = reshape(saeOpt(1:modelParas{i}.hiddenSize*modelParas{i}.inputSize),modelParas{i}.hiddenSize,modelParas{i}.inputSize);
		stack{i}.b = saeOpt(2*modelParas{i}.hiddenSize*modelParas{i}.inputSize+1:2*modelParas{i}.hiddenSize*modelParas{i}.inputSize+modelParas{i}.hiddenSize);
	end
	[stackparas,netconfig] = stack2params(stack);
	stackedAETheta = [softmaxOptTheta;stackparas];
end
