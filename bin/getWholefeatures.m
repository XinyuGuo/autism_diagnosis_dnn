function [newfeatures,newbias]= getWholefeatures()
	addpath '../data/';	
	saes_num = 5;
	vsize = combntns(116,2); 
	hsize = 50;
	load('trainingset','traindata');	
	[newfeatures,newbias]=gethiddenfeatures(saes_num,hsize,vsize,traindata);	
end
