function [highestfishernodes] = getLayerFisher()
%% get mean fisher score for each layer and its std
	load('../data/data.mat','data');
	load('../data/labels.mat','labels');
	load('../data/optParas.mat','stackedAEOptTheta');		
	load('../data/netconfig.mat','netconfig');
	load('../data/modelParas.mat','modelParas');
	
	index = size(modelParas,1)-1;
	numoflayers = size(modelParas,1)-2;
	softparas = modelParas{index};
	hiddenSize = softparas.inputSize;
	numClasses = softparas.numClasses;
	stack = params2stack(stackedAEOptTheta(hiddenSize*numClasses+1:end), netconfig);
    
    highestfishernodes = cell(3,1);
    nodesperlayer = 3;
    for i=1:numoflayers		
		if i ==1
			[B,I]= fisherscore(stack{i}.w,data',labels);
			%mean(B)	
		elseif i==2
		 	z= bsxfun(@plus,stack{i-1}.w*data',stack{i-1}.b);
			[B,I]= fisherscore(stack{i}.w,z,labels);
			%mean(B)
		else
		 	z= bsxfun(@plus,stack{i-2}.w*data',stack{i-2}.b);
			a= sigmoid(z);
		 	z= bsxfun(@plus,stack{i-1}.w*a,stack{i-1}.b);
			[B,I]= fisherscore(stack{i}.w,z,labels);
			%mean(B)
        end		
        weights = stack{i}.w;	
        highestfishernodes{i} = weights(I(1:nodesperlayer),:);
	end
end

function [sigm] = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
