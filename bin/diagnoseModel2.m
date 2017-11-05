function diagnoseModel2()
	%load data and labels
	datapath ='../data/data.mat';
	labelspath='../data/labels.mat';
    load(labelspath,'labels');
	load(datapath,'data');
	
	paras = generateParas(size(data,2));
	numOfsae = paras{18,1};
	fselec = true;
	[trainlabels,traindata,testlabels,testdata]	= getOriFeatureSets(numOfsae,paras);
	%if fselec
		[beforeTunepred,afterTunepred]=deepFramework(trainlabels,traindata',testlabels,testdata',paras,fselec);
	%else
	fselec = false;
		[beforeTunepred,afterTunepred]=deepFramework(trainlabels,traindata',testlabels,testdata',paras,fselec);
	%end
end
