function [softmaxOptTheta] = trainSoftmax(softmaxParas,saefeatures,saeflabels)
	saeSoftmaxTheta = 0.005*randn(softmaxParas.inputSize*softmaxParas.numClasses,1);
	softmaxModel = softmaxTrain(softmaxParas.inputSize,softmaxParas.numClasses,softmaxParas.lambda,saefeatures,saeflabels,softmaxParas.options);
	softmaxOptTheta = softmaxModel.optTheta(:);
end
