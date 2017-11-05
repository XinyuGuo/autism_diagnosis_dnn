function [folds_info,test_fold_num] = trainDeepFramework(saenum)
	disp('trainDeepmodel');
	%load parameters and data
	datapath ='../data/data.mat';
	labelspath='../data/labels.mat';
	load(labelspath,'labels'); 		
	load(datapath,'data');
	datadim = size(data,2);
	numofsae= saenum;		
    featureSelection = false;
	modelParas = generateLayerParas(datadim,numofsae,featureSelection);
	save('../data/modelParas.mat','modelParas')
	%celldisp(modelParas);

	%divid data into folds		
	%folds_num = 5;
	%folds_info=getCrossvalidation(folds_num);
	%test_fold_num = randi(folds_num);
	%[trainlabels,traindata,testlabels,testdata] = getDataforFold(test_fold_num,folds_info,folds_num,data,labels);
	
    %data for feature analysis
    trainlabels = labels;
	traindata = data; 

	%train the diagnose model - sparse AE
	layernum = size(modelParas,1);
	fprintf('parameter sets num is %d\n',layernum);
	train = traindata';
	layerOpt = cell(numofsae,1);
	for i = 1:numofsae
		fprintf('train layer %d: \n',i);
		[saeOptTheta] = trainSAE(modelParas{i},train);
		layerOpt{i,1} = saeOptTheta;
		train = feedForwardAutoencoder(saeOptTheta,modelParas{i}.hiddenSize,modelParas{i}.inputSize,train); 		
	end

	%train the diagnose model - softmax	
	softmaxParas = modelParas{numofsae+1};
	softmaxParas.numClasses
	trainlabels = double(trainlabels);
	softmaxOptTheta = trainSoftmax(softmaxParas,train,trainlabels);	
	%train the diagnose mdoel - fine tune	
	[stackedAETheta,netconfig] = getFinetuneParas(modelParas,layerOpt,softmaxOptTheta); 
	save('../data/netconfig.mat','netconfig');
	[stackedAEOptTheta]=finetuneDeepModel(stackedAETheta,netconfig,modelParas,traindata',trainlabels);
    save('../data/optParas.mat','stackedAEOptTheta');	
	%test model
	%[pred_notune] = modelpredict(stackedAETheta,modelParas,netconfig,testdata');
	%beforeTunepred = mean(testlabels(:) == pred_notune(:));
	%fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);

	%[pred_tune] = modelpredict(stackedAEOptTheta,modelParas, netconfig, testdata');
	%afterTunepred = mean(testlabels(:) == pred_tune(:));
	%fprintf('After Finetuning Test Accuracy: %0.3f%%\n', afterTunepred * 100);
end
