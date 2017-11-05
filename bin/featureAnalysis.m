function featureAnalysis() 
	layerNum = 2;
	toplinks = 5;
	hidden = 50;  
	visible = 6670;
	features = getLatentfeatures(layerNum,toplinks,hidden,visible);
	%size(features{1,1})
	%size(features{2,1})
	max(max(features{1,1}))
	max(max(features{2,1}))
end
