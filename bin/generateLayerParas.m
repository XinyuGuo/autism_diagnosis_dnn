function [modelParas] = generateLayerParas(datadim,layernum,fslect)
	LayerParas = cell(7,1);
	% first layer parameters 
	LayerParas{1}.inputSize = datadim;
	LayerParas{1}.hiddenSize= 50;
	LayerParas{1}.saeTheta=initializeParameters(LayerParas{1}.hiddenSize,LayerParas{1}.inputSize);
	LayerParas{1}.sparsityParam=0.05;
	LayerParas{1}.lambda=3e-3;
	LayerParas{1}.beta=3;

	opt.Method = 'lbfgs';	

    if fslect
	    opt.MaxIter= 80;	
    else
        opt.MaxIter= 400;
    end

	opt.display = 'on';
	LayerParas{1}.options=opt;
		
	% second layer parameters 
	LayerParas{2}.inputSize = LayerParas{1}.hiddenSize;
	LayerParas{2}.hiddenSize= 50;
	LayerParas{2}.saeTheta=initializeParameters(LayerParas{2}.hiddenSize,LayerParas{2}.inputSize);
	LayerParas{2}.sparsityParam=0.05;
	LayerParas{2}.lambda=3e-3;
	LayerParas{2}.beta=3;

	opt.Method = 'lbfgs';	
	opt.MaxIter= 400;	
	opt.display = 'on';
	LayerParas{2}.options=opt;

	% third layer parameters 
	LayerParas{3}.inputSize = LayerParas{2}.hiddenSize;
	LayerParas{3}.hiddenSize= 50;
	LayerParas{3}.saeTheta=initializeParameters(LayerParas{3}.hiddenSize,LayerParas{3}.inputSize);
	LayerParas{3}.sparsityParam=0.05;
	LayerParas{3}.lambda=3e-3;
	LayerParas{3}.beta=3;

	opt.Method = 'lbfgs';	
	opt.MaxIter= 400;	
	opt.display = 'on';
	LayerParas{3}.options=opt;

	% fourth layer parameters
	LayerParas{4}.inputSize = LayerParas{3}.hiddenSize;
	LayerParas{4}.hiddenSize= 50;
	LayerParas{4}.saeTheta=initializeParameters(LayerParas{4}.hiddenSize,LayerParas{4}.inputSize);
	LayerParas{4}.sparsityParam=0.05;
	LayerParas{4}.lambda=3e-3;
	LayerParas{4}.beta=3;

	opt.Method = 'lbfgs';	
	opt.MaxIter= 4;	
	opt.display = 'on';
	LayerParas{4}.options=opt;

	% fifth layer parameters
	LayerParas{5}.inputSize = LayerParas{4}.hiddenSize;
	LayerParas{5}.hiddenSize= 50;
	LayerParas{5}.saeTheta=initializeParameters(LayerParas{5}.hiddenSize,LayerParas{5}.inputSize);
	LayerParas{5}.sparsityParam=0.05;
	LayerParas{5}.lambda=3e-3;
	LayerParas{5}.beta=3;

	opt.Method = 'lbfgs';	
	opt.MaxIter= 4;	
	opt.display = 'on';
	LayerParas{5}.options=opt;
	
	% softmax layer parameters
	LayerParas{6}.lambda = 3e-3;
	LayerParas{6}.numClasses = 2;
	LayerParas{6}.inputSize = LayerParas{layernum}.hiddenSize;

	opt.Method = 'lbfgs';	
	opt.MaxIter= 100;	
	opt.display = 'on';
	LayerParas{6}.options=opt;

	% fine-tune parameters
	LayerParas{7}.lambda = 1e-4;
	LayerParas{7}.numClasses= 2;
	LayerParas{7}.saeSoftmaxTheta = 0.005*randn(LayerParas{layernum}.hiddenSize*LayerParas{7}.numClasses,1);
	
	opt.Method = 'lbfgs';
	opt.maxIter = 400;
	opt.display = 'on';
	LayerParas{7}.options = opt;
	
	% construct parameters for the deep framework
	modelParas = cell(layernum+2,1);	
	for i = 1:layernum
		modelParas{i} = LayerParas{i};
	end
	modelParas{layernum+1} = LayerParas{6};
	modelParas{layernum+2} = LayerParas{7};
end
