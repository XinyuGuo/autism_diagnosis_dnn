function [folds_info]=testnested()
	datapath ='../data/data.mat';
	labelspath='../data/labels.mat';
    load(labelspath,'labels'); 		
	load(datapath,'data');

	folds_num = 5;
	folds_info=getCrossvalidation(folds_num);
	
	%for i = 1 : folds_num % outer loop cross-validation
		[trainlabels,traindata,testlabels,testdata] = getDataforFold(1,folds_info,folds_num,data,labels);
	%end
end
