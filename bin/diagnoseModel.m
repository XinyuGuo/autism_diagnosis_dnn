function diagnoseModel()
	%load data and labels
	datapath ='../data/data.mat';
	labelspath='../data/labels.mat';
    load(labelspath,'labels');
	load(datapath,'data');
	%divide labels into folds_num partas
	folds_num = 5;
	folds_info=getCrossvalidation(folds_num);
	%in the test case, run the simulation only for one fold
	test = true;
	compare = true;	
	if test == true
		sparsity_grid = false;
		feature_selection = false;
		test_fold_num=randi(folds_num);	
		[trainlabels,traindata,testlabels,testdata] = getDataforFold(test_fold_num,folds_info,folds_num,data,labels);
		%set up model parameters
		paras = generateParas(size(data,2),sparsity_grid);
		deepFramework(trainlabels,traindata',testlabels,testdata',paras,feature_selection);
	else % nested cross-validation training:validation:test 3:1:1
		tic;
		sparsity_grid = true;
		feature_selection = false;
		[paras,sparsity_array] = generateParas(size(data,2),sparsity_grid);
		final = [];
		I = [];
		for i = 1 : folds_num % outer loop cross-validation
			fprintf('*******************************************\n');
			fprintf('OUTER LOOP: fold %d\n',i);
			fprintf('*******************************************\n');
			[trainlabels,traindata,testlabels,testdata] = getDataforFold(i,folds_info,folds_num,data,labels);
			inner_folds_info = folds_info;% get rid of the test fold 
			inner_folds_info(i,:,:) = [];
			for j = 1:size(sparsity_array,2) % grid search
			    fprintf('Sparsity Parameter Value: %d\n',sparsity_array(1,j));	 	
				pred_sparsity = cell(size(sparsity_array,2),1); %matrix.................................
			   	v_labels = [];	
				t_labels = [];
				v_pred = [];
				t_pred = [];
				for k = 1:folds_num-1% inner crossvalidation
					fprintf('INNER LOOP: fold %d\n',k);
					[trainlabels,traindata,validationlabels,validationdata] = getDataforFold(k,inner_folds_info,folds_num-1,data,labels);
					[vpred_notune,tpred_notune,vpred_tune,tpred_tune]=deepFramework_batch(trainlabels,traindata',validationlabels,validationdata',testlabels,testdata',paras,feature_selection,sparsity_grid,sparsity_array(1,j));
			   		v_labels = [v_labels,validationlabels];	
					%size(v_labels)
					t_labels = [t_labels,testlabels];
					%size(t_labels)
					v_pred = [v_pred,vpred_tune];
					%size(v_pred)
					t_pred = [t_pred,tpred_tune];
					%size(t_pred)
				end
				fprintf('\n');
				predvalues = zeros(1,2);
				predvalues(1,1)=mean(t_labels(:)==t_pred(:));
				predvalues(1,2)=mean(v_labels(:)==v_pred(:));
				pre_sparsity{j,1}=predvalues; %matrix...............................................
			end
			filepath= strcat('../result/validation_result_sparsity_',num2str(i),'.mat');
		    save(filepath,'pre_sparsity');
		    final_result = cell2mat(pre_sparsity);
			[d,i] = max(final_result(:,2));
			I = [I,i];
			final_test_result = final_result(1,i);
			final = [final,final_test_result]
		end
		save('../result/test_result_sparsity.mat','final');	
		save('../result/which_sparsity.mat','I');	
		fprintf('final accuracy is %f',mean(final));
		toc;
	end
end
