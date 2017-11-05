% % % % % % % % % %
% Project: Autism
% Author: Xinyu Guo
% Date: 6-9-2016
% train numOfsae sparse auto-encoders and save its weights
% usage : getOriFeatureSets(5)
% % % % % % % % % % 

function [trainlabels,traindata,testlabels,testdata]=getOriFeatureSets(numOfsae,paras)
	% set parameters for the SAE
	saeParas = cell(5,1);		
	saeParas{1,1}= paras{1,1}; 		% visible  
	saeParas{2,1}= paras{2,1};      % hidden 
	saeParas{3,1}= paras{4,1};      % sparsityParam 
	saeParas{4,1}= paras{5,1};      % lambda    
	saeParas{5,1}= paras{6,1};      % beta          
	% set parameters for the optimization function	
	optParas=cell(3,1);
	optParas{1,1} = paras{8,1};     % optimization method 
	optParas{2,1} = 10;				% max training times
	optParas{3,1} = paras{10,1};	% display or not
	% get the training data	
	datapath = '../data/data.mat';		
	labelspath = '../data/labels.mat';
	load(datapath,'data');		
	load(labelspath,'labels');
	folds_num = 5;
	folds_info = getCrossvalidation(folds_num);
	test_fold_num = randi(folds_num);	
	[trainlabels,traindata,testlabels,testdata]=getDataforFold(test_fold_num,folds_info,folds_num,data,labels);

	num = numOfsae;
	for i = 1:numOfsae
		filepath= getFilepath(i);
		opttheta = getFeaturesFromSAE(saeParas,optParas,traindata);		
		save(filepath,'opttheta');
	end
end

function [filepath] = getFilepath(i)%get filename for storing opttheta
	filedir = '../data/';
	filepath= strcat(filedir,'opttheta',num2str(i),'.mat');
end
