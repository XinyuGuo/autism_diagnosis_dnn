function test()
	%addpath '../data/';
	%load('data.mat','data')
	%fig=figure;	
	%imagesc(data);
	%data(126,6670)
	%val = sum(traindata'==1)	
	%indices = [1 2 3 4 5 6 7 8 9 10];
	%for i = 1:size(indices,2)
	%	[r,c]=getIndices(i);
	%	disp(indices(i));
	%	disp([r, c]);
	%end
	[r,c]=getIndices(30815)
end

function [row,col]= getIndices(pos)
	len=250-1;
	temp_len = 0;
	col = 1;
	notFound = true;

	while notFound&&len>0
		temp_len=temp_len+len;
		if pos>temp_len
			len=len-1;
			col=col+1;
		else		
			row =250-(temp_len-pos);
			notFound=false;
		end
	end
end
