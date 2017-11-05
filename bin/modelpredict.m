function [pred] = modelpredict(theta,modelparas,netconfig,data)
	%extract parameters
	index = size(modelparas,1)-1;
	softparas = modelparas{index};
	hiddenSize = softparas.inputSize;
	numClasses = softparas.numClasses;

	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
	stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
	%feed forward
	a = data;
	for i = 1:numel(stack)
		z = bsxfun(@plus,stack{i}.w*a,stack{i}.b);
		a = sigmoid(z);
	end
	a_softmax = exp(softmaxTheta*a);
	a_softmax = bsxfun(@rdivide,a_softmax,sum(a_softmax));
	[p,pred] = max(a_softmax,[],1);
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
