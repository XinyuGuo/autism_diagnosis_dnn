% % % % % % % % % % % %
% date : 6-8-2016
% author: Xinyu Guo
% Group analyze average the whole brain FCs between Autism and TDC
% generate the 2D lower trianglar matrix for each subject whole brain FC patterns.
% % % % % % % % % % % %

function groupFcanalysis()
	numOfregions = 116;
	datapath='../data/data.mat';
	labelpath = '../data/labels.mat';
	load(datapath,'data');	
	load(labelpath,'labels');	
	singlebrain = true;

	if singlebrain
		subnum = size(data,1);
		matrix = triu(ones(numOfregions),1)';
		matrices = repmat(matrix,[subnum,1]);
		matrices = permute(reshape(matrices',[numOfregions,numOfregions,subnum ]),[2,1,3]);
		matrices(~~matrices) = data'; 
	end

	patients= find(labels==1);
	tdc= find(labels==2);
	pdata = data(patients,:);
	tdata =	data(tdc,:);
			
	pvalue = zeros(1,size(data,2));
    statsvalue = zeros(1,size(data,2));

	for i=1:size(data,2)		
		[h,p,ci,stats] = ttest2(pdata(:,i),tdata(:,i));	
	   	pvalue(i) = p; 						
        statsvalue(i) = stats.tstat;
	end
	patients_mean = mean(pdata);
	tdc_mean = mean(tdata);

    %value = sort(pvalue);
    %value(150)
    %pvalue(pvalue>0.0114)=0;
	%matrix = getSymmatrix(numOfregions,pvalue);

	matrix1 = getSymmatrix(numOfregions,statsvalue);
	matrix2 = getSymmatrix(numOfregions,patients_mean);
	matrix3 = getSymmatrix(numOfregions,tdc_mean);
		
    %fig = figure;
    %imagesc(matrix)

	fig = figure;	
	imagesc(matrix1);
	fig = figure;	
	imagesc(matrix2);
	fig = figure;	
	imagesc(matrix3);
	
	% for testing getSymmatrix
	%a = [1,2,3,4,5,6,7,8,9,10];
	%b = getSymmatrix(5,a);
end

function [symmetric] = getSymmatrix(dim,data)
    M= triu(ones(dim),1)';
	M(~~M) = data;
	sym_indice = triu(true(size(M)),1)';		
	symmetric = M';
	symmetric(sym_indice)=data;
end
