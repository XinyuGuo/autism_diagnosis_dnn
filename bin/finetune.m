function [cost,grad] = finetune(theta,modelparas,netconfig,data,labels)
	numofsae = size(modelparas,1)-2;	
	hiddenSize = modelparas{numofsae+1}.inputSize;
	numClasses= modelparas{numofsae+1}.numClasses;
	softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
	stack = params2stack(theta(hiddenSize*numClasses+1:end),netconfig);
	
	
	softmaxThetaGrad = zeros(size(softmaxTheta));
	stackgrad = cell(size(stack));
	for d = 1:numel(stack)
    	stackgrad{d}.w = zeros(size(stack{d}.w));
    	stackgrad{d}.b = zeros(size(stack{d}.b));
	end

	cost = 0;
	M = size(data,2); 
	groundTruth = full(sparse(labels,1:M,1));
	a = data;
	
	%feed forward process
	saesum = cell(numofsae,1);
	saeactivation = cell(numofsae,1);
	for i = 1:numofsae
		z = bsxfun(@plus,stack{i}.w*a,stack{i}.b);
		a = sigmoid(z);
		saesum{i} = z;
		saeactivation{i} = a;
	end
	z_soft = softmaxTheta * a;
	a_soft = exp(z_soft);
	a_soft = bsxfun(@rdivide,a_soft,sum(a_soft));

	%backpropagation	
	delta_soft = -(groundTruth - a_soft);
	delta = (softmaxTheta' * delta_soft) .* sigmoidGrad(saesum{numofsae});
	softmaxThetaGrad = -(1. / M)*(groundTruth-a_soft)*saeactivation{numofsae}'+modelparas{numofsae+2}.lambda*softmaxTheta; 

	for i = numel(stack):-1:1
		if i == 1
			layeract = data';
		else
			layeract = saeactivation{i-1}';
		end

		stackgrad{i}.w = (1. / M)*delta*layeract+modelparas{numofsae+2}.lambda*stack{i}.w; 
		stackgrad{i}.b = (1. / M)*sum(delta, 2); 

		if i>1	
			delta =(stack{i}.w'*delta).*sigmoidGrad(saesum{i-1});
		end
	end	
	
	sum_w = 0;
	for i=1:numel(stack) 			
		sum_w = sum_w+sum(sum(stack{i}.w.^2));
	end

	cost = -(1./ M)*sum(sum(groundTruth.*log(a_soft)))+(modelparas{numofsae+2}.lambda/2.)*sum(sum(softmaxTheta.^2))+(modelparas{numofsae+2}.lambda/2.)*sum_w; 

	function grad = softmaxGrad(x) 
		e_x = exp(-x); 
		grad = e_x ./ (1 + (1-e_x).*e_x ).^2; 
	end

	function grad = sigmoidGrad(x)
		e_x = exp(-x); 
		grad = e_x ./ ((1 + e_x).^2);  
	end  

	function sigm = sigmoid(x)
		sigm = 1 ./ (1 + exp(-x));
	end
	
	%% Roll gradient vector
	grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];
	
end
