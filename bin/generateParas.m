function [paras,sparsityParam_sae]=generateParas(inputsize,grid_sparsity)
	paras= cell(18,1);
	inputSize = inputsize;
	paras{1,1} = inputSize;
	hiddenSizeL1 = 200;		
	paras{2,1} = hiddenSizeL1; 
	hiddenSizeL2 = 200;% grid search		
	paras{3,1} = hiddenSizeL2;%grid search

	if grid_sparsity%no grid search
		sparsityPArray = [0.01,0.03,0.05,0.07,0.09];
		sparsityParam_sae = sparsityPArray; 
	else%grid search
		sparsityParam_sae = 0.05;
	end

	paras{4,1} = sparsityParam_sae;
	lambda_sae = 3e-3;
	paras{5,1} = lambda_sae;
	beta_sae = 3;
	paras{6,1} = beta_sae;
	numClasses = 2;
	paras{7,1} = numClasses;
	
	%train SAE options	
	trainmethod = 'lbfgs';	
	paras{8,1} = trainmethod;
	maxtraintimes_sae = 40;
	paras{9,1} = maxtraintimes_sae;
	traindisplay = 'on';
	paras{10,1} = traindisplay;
	traincorr = 10;	
	paras{11,1} = traincorr;

	%softmax parameters
	lambda_soft = 3e-3;
	paras{12,1} = lambda_soft;
	maxtraintimes_soft = 100;
	paras{13,1} = maxtraintimes_soft;

	%fine-tuning parameters
	lambda_finetune = 1e-4;
	paras{14,1} = lambda_finetune;
	maxtraintimes_finetune = 40;
	paras{15,1} = maxtraintimes_finetune;
	traindisplay_finetune = 'on';
	paras{16,1} = traindisplay_finetune;
	trainmethod_finetune = 'lbfgs';
	paras{17,1}	= trainmethod_finetune;
	
	saes_num = 5;
	paras{18,1} = saes_num;
end
