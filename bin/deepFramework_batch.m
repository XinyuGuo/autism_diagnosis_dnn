function [vpred_notune,tpred_notune,vpred_tune,tpred_tune] = deepFramework_batch(trainlabels,train,testlabels,test,testlabels2,test2,paras,fselec,s_grid,sparsity)
% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

% the follwoing are parameters of SAE.
inputSize = paras{1,1};
hiddenSizeL1 = paras{2,1};
hiddenSizeL2 = paras{3,1}; 

if s_grid % grid search sparsity parameters
	sparsityParam_sae = sparsity;
else
	sparsityParam_sae = paras{4,1};  
end

lambda_sae = paras{5,1};
beta_sae = paras{6,1};  

%% STEP 1 : Train  the first sparse autoencoder
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

[cost, grad] = sparseAutoencoderCost(sae1Theta, inputSize, hiddenSizeL1,lambda_sae,sparsityParam_sae, beta_sae, train);
                                 
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = paras{8,1}; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = paras{9,1};	  % Maximum number of iterations of L-BFGS to run 
options.display = paras{10,1};
%options.Corr = paras{11,1};
tic;
if ~fselec
	disp('train first layer:');
	[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                    inputSize, hiddenSizeL1, ...
                                    lambda_sae, sparsityParam_sae, ...
                                    beta_sae, train), ...
                                    sae1Theta, options);
	filters = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize),hiddenSizeL1,inputSize);
	save('../data/filters1.mat','filters');
else
	disp('feature selction for the first layer:');
    [newfeatures,newbias] = gethiddenfeatures(paras{18,1},paras{2,1},paras{1,1},train);
    newf = reshape(newfeatures,inputSize*hiddenSizeL1,1);
	newbias2 = zeros(paras{1,1},1);
	sae1OptTheta=[newf(:);newf(:);newbias(:);newbias2];	
end
%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.

sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1,inputSize, train);
[cost, grad] = sparseAutoencoderCost(sae2Theta, hiddenSizeL1, hiddenSizeL2,lambda_sae,sparsityParam_sae,beta_sae,sae1Features);
%%  Use minFunc to minimize the function
addpath minFunc/
options.Method = paras{8,1}; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = paras{9,1};	  % Maximum number of iterations of L-BFGS to run 
options.display = paras{10,1};

disp('train second layer:');
[sae2OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(p, ...
                               hiddenSizeL1, hiddenSizeL2, ...
                               lambda_sae, sparsityParam_sae, ...
                               beta_sae, sae1Features), ...
                               sae2Theta, options);

filters = reshape(sae2OptTheta(1:hiddenSizeL1*hiddenSizeL2),hiddenSizeL1,hiddenSizeL2);
save('../data/filters2.mat','filters');

%%======================================================================
%% STEP 4: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2,hiddenSizeL1,sae1Features);
%  Randomly initialize the parameters
numClasses = paras{7,1}; 
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
%
Options.maxIter = paras{13,1};
Options.display = 'off';
lambda_soft = paras{12,1}; 
trainlabels = double(trainlabels);
disp('train softmax layer');
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda_soft, ...
                            sae2Features, trainlabels, Options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
%% -------------------------------------------------------------------------
%
%
%%%======================================================================
%%% STEP 5: Finetune softmax model
%
%% Implement the stackedAECost to give the combined cost of the whole model
%% then run this cell.
%
% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".


addpath minFunc/ 
lambda_finetune = paras{14,1}; 
options.Method = paras{17,1};  
options.maxIter = paras{15,1};	  
options.display = paras{16,1};
disp('fine tune');

[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ... 
                                     numClasses, netconfig, lambda_finetune, ... 
                                     train, trainlabels), stackedAETheta, options); 
toc;
%% -------------------------------------------------------------------------
%
%%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%
%
% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
 
% testLabels(testLabels == 0) = 10; % Remap 0 to 10
[vpred_notune] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, test);

[tpred_notune] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, test2);

%beforeTunepred = mean(testlabels(:) == pred(:));
%fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);
%
[vpred_tune] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                              numClasses, netconfig, test);
%size(vpred_tune);
[tpred_tune] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                              numClasses, netconfig, test2);
%size(tpred_tune);
vafterTunepred = mean(testlabels(:) == vpred_tune(:));
tafterTunepred = mean(testlabels2(:) == tpred_tune(:));
fprintf('After Finetuning Validation Accuracy: %0.3f%%\n', vafterTunepred * 100);
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', tafterTunepred * 100);
fprintf('\n');
end
%%fprintf(fid,'Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);
%%s = ' ';
%%fprintf(fid,'%s',s);
%%fprintf(fid,'After Finetuning Accuracy: %0.3f%%\n', afterTunepred * 100);
%%
%
%if Results_To_File
%fid=fopen('result.txt','w');
%fprintf(fid,'%f',beforeTunepred);
%fprintf(fid,'%f',afterTunepred);
%s = ' ';
%fprintf(fid,'%s',s);
%fclose(fid);
%end;
%
%end
