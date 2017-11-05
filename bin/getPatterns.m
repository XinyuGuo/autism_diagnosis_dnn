function  getPatterns()
	load('../data/data.mat','data');
	load('../data/labels.mat','labels');
	load('../data/optParas.mat','stackedAEOptTheta');		
	load('../data/netconfig.mat','netconfig');
	load('../data/modelParas.mat','modelParas');

    highestfishernodes = getLayerFisher();
    
    index = size(modelParas,1)-1;
	softparas = modelParas{index};
	hiddenSize = softparas.inputSize;
	numClasses = softparas.numClasses;
    stack = params2stack(stackedAEOptTheta(hiddenSize*numClasses+1:end), netconfig);
    
	topNconnections_1 = 20; % first hidden layer 
    topNconnections_2 = 5; % second hidden layer
    topNconnections_3 = 3; % third hidden layer
	
	indices= cell(3,1);
    values = cell(3,1); 
    for i= 1:3 % 3 layers 
         if i ==1 % the first hidden layer
             weights = highestfishernodes{i}; % 3 * 6670
             [B1,I1] = sort(weights,2,'descend');
             indices{i} = I1(:,1:topNconnections_1); % 3*20
             values{i} = B1 (:,1:topNconnections_1);
         elseif i==2 % the second hidden layer
             weights = highestfishernodes{i};
             [B2,I2] = sort(weights,2,'descend');
             top_weightsid = I2(:,1:topNconnections_2);
             layer1weights = stack{i-1}.w;
             indices2 = [];
             values2 = [];
             for j = 1:3 % 3 nodes
                w = layer1weights(top_weightsid(1,:),:);
                [B,I] = sort(w,2,'descend');
                %size(I(:,1:topNconnections_1));
                currIndices = I(:,1:topNconnections_1);
                currValues = B(:,1:topNconnections_1);
                indices2 = [indices2;currIndices(:)'];
                values2 = [values2;currValues(:)'];
             end
             indices{i} = indices2;
             size(indices{i}); % 
             values{i} = values2;
             size(indices);
         else % the third hidden layer 
            weights = highestfishernodes{i};
            size(weights);
            [B3,I3] = sort(weights,2,'descend'); 
            size(I3); % 3*50
            top_weightsid = I3(:,1:topNconnections_3);
            size(top_weightsid); %3*3
            layer2weights = stack{i-1}.w;
            indices3 = [];
            for j = 1:3 % 3 nodes
                weights2 = layer2weights(top_weightsid(j,:),:); % 3*50
                [B2,I2] = sort(weights2,2,'descend');
                top_weightsid_2 = I2(:,1:topNconnections_2); %3*5
                indices3 = [indices3;top_weightsid_2(:)'];
            end
            size(indices3); %3*15
            
            layer3weights = stack{i-2}.w;
            indices4=[];
            value4 =[];
            for k = 1:3 
                weights3 = layer3weights(indices3(k,:),:);
                size(weights3); % 15*6670
                [B3,I3] = sort(weights3,2,'descend');
                currIndices = I3(:,1:topNconnections_1);
                currValue = B3(:,1:topNconnections_1);
                indices4 = [indices4;currIndices(:)'];
                value4 = [value4;currValue(:)'];
                size(currIndices); %15*20
            end
            size(indices4); %3*300
            %indices4 
            indices{i} = indices4;
            size(indices);
            values{i} = value4; 
         end
    end
    size(indices{1});%3*20
    size(indices{2})%3*30
    size(indices{3})%3*100
    size(values{1})%3*20
    size(values{2})%3*30
    size(values{3})%3*100
    
    
    ids = indices{3};
    value = values{3};
    size(ids);
    for n = 1:3
        patterns = zeros(1,6670);
        patterns(ids(n,:))=1;%value(n,:);
        %sum(patterns);
        patternMatrix = getSymmatrix(116,patterns);
        filename = strcat('../data/patternMatrix3_',num2str(n),'.mat');
        %filename;
        save(filename,'patternMatrix');
    end


%   patterns = zeros(1,6670);
% 	patterns(indices) = 1;
% 	indices
% 	sum(patterns)	
% 	patternMatrix = getSymmatrix(116,patterns);
% 	save('../data/patternMatrix.mat','patternMatrix');
end

function [symmetric] = getSymmatrix(dim,data)
    M= triu(ones(dim),1)';
	M(~~M) = data;
	sym_indice = triu(true(size(M)),1)';		
	symmetric = M';
	symmetric(sym_indice)=data;
end