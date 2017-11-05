% % % % % % % % % % %
% Project: Autism
% Author: Xiny Guo
% Date: 6-1-2016
% generating the cross validation scheme
% usage : getCrossvalidation(5)
% % % % % % % % % % %

function [folds] = getCrossvalidation(fold)
	datapath ='../data/data.mat';
	labelspath='../data/labels.mat';
    load(labelspath,'labels'); 		
	load(datapath,'data');
	labels_patients= find(labels==1);
	labels_tdc = find(labels==2);	
	patients_num = size(labels_patients,2);
	tdc_num = size(labels_tdc,2);
	patients_left_num =mod(patients_num,fold);	
	tdc_left_num =mod(tdc_num,fold);	
	p_fold_num =floor(patients_num/fold);
	tdc_fold_num = floor(tdc_num/fold);		
						
	p_labels_fold=cell(fold,1);	
	tdc_labels_fold=cell(fold,1);
	patients_labels_mixed = randperm(patients_num);
	tdc_labels_mixed = randperm(tdc_num);

	for i = 1:fold				
		p_labels_fold{i,1}=labels_patients(patients_labels_mixed(1+(i-1)*p_fold_num:i*p_fold_num));
		tdc_labels_fold{i,1}=labels_tdc(tdc_labels_mixed(1+(i-1)*tdc_fold_num:i*tdc_fold_num));
	end
	
	for i = 1:patients_left_num
		p_index = labels_patients(patients_labels_mixed(p_fold_num*fold+i));
		p_labels_fold{i,1}=[p_labels_fold{i,1},p_index];
	end	

	for i = 1:tdc_left_num
		tdc_index = labels_tdc(tdc_labels_mixed(tdc_fold_num*fold+i));
		tdc_labels_fold{i,1}=[tdc_labels_fold{i,1},tdc_index];
	end	

	folds = cell(fold,2,1);
	for i = 1:fold
		thisfold=cell(2,1);
		%fprintf('this is patient %d fold:\n',i);
		%disp(p_labels_fold{i,1});
		%fprintf('this is health  %d fold:\n',i);
		%disp(tdc_labels_fold{i,1});
		%fprintf('************************\n');
		folds{i,1,1}=p_labels_fold{i,1};%patients labels
		folds{i,2,1}=tdc_labels_fold{i,1};%tdc labels
	end		
end	
