function [features]=getLatentfeatures(layerNum,toplinks,hidden,visibel)
	originalfeatures = cell(layerNum,1);
	datapath = '../data/';

	for layer=1:layerNum
		filterspath = strcat(datapath,'filters',num2str(layer),'.mat');			
		load(filterspath,'filters');
		originalfeatures{layer,1} = filters;
	end
	
	features = cell(layerNum,1);
	features{1,1} = originalfeatures{1,1};

	for layer=2:layerNum
		latentfeatures = [];	
		highlf= originalfeatures{layer,1};
		lowlf = features{layer-1,1}; 
		for i = 1:hidden
			[B,I]=sort(highlf(i,:),'descend');
			indices = I(1:toplinks);	
			weightsvalue = B(1:toplinks);
			latentfeatures = [latentfeatures;weightsvalue*lowlf(indices,:)];
		end
		features{layer,1} = latentfeatures;
	end
end
