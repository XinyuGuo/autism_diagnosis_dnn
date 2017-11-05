% % % % % % % % % %
% Project: Autism
% Author: Xinyu Guo
% Date: 6-9-2016
% triggering training a number of auto-encoders 
% saving the training data
% usage: trainSaes()
% % % % % % % % % %

function trainSaes()	
	datapath =  '../data/';
	numOfsae = 5;
	trainpath = strcat(datapath,'trainingset.mat');
	[~,traindata,~,~] = getOriFeatureSets(numOfsae);
	save(trainpath,'traindata');
end
