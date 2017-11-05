function [trainlabels,train,testlabels,test]=getDataforFold(fold_num,folds_info,folds_num,data,labels)
	test_labels = [folds_info{fold_num,1,1},folds_info{fold_num,2,1}];
	test = data(test_labels,:);
	testlabels = labels(test_labels);
	train_labels = [];
	for i = 1:folds_num	
		if i == fold_num
			continue;
		end
		this_train_labels = [folds_info{i,1,1},folds_info{i,2,1}];
		train_labels=[train_labels,this_train_labels];
	end
	train= data(train_labels,:);
	trainlabels = labels(train_labels);
end
