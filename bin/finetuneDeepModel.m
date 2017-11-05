function [stackedAEOptTheta]=finetuneDeepModel(stackedAETheta,netcofig,modelParas,train,trainlabels)
	addpath minFunc/
	numofparasets = size(modelParas,1);
	[stackedAEOptTheta,cost] = minFunc(@(p)finetune(p,modelParas,netcofig,train,trainlabels),stackedAETheta,modelParas{numofparasets}.options);	
end
