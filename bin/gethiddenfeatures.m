function [newfeatures,newbias]=gethiddenfeatures(numOfsaes,hidden,visible,traindata)
    addpath '../data/';
	F = [];
	B = [];
	for i=1:numOfsaes
		filename = getfilename(i);
		load(filename,'opttheta');		
		W1 = reshape(opttheta(1:hidden*visible), hidden, visible);
		b1 = opttheta(2*hidden*visible+1:2*hidden*visible+hidden);
		F = cat(1,F,W1);
	    B = cat(2,B,b1); 	
	end
	D = pdist(F);
	%size(find(D>33))
	threshold = 16;
	%newfeatures = getNewfeatures(numOfsaes,hidden,D,threshold,F);
	newfeatures = getMostdispairs(numOfsaes,hidden,D,F);
	newbias = mean(B,2);
	%size(newfeatures)	
	%DD = pdist(newfeatures)
end

function [newfeatures] =getMostdispairs(numOfsaes,hidden,D,F)
	[B,I]=sort(D,'descend');					
	f_num = 0;		
	count = 1;
	mapObj = containers.Map('KeyType','int32','ValueType','int32');

	while f_num<hidden && count<size(I,2)
		[r,c]=getIndices(I(count),numOfsaes,hidden);
		if ~isKey(mapObj,r)
			mapObj(r)=r;
			f_num=f_num+1;
		end

		if ~isKey(mapObj,c)
			mapObj(c)=c;
			f_num=f_num+1;
		end
		count = count+1;
	end
	
	keySet = keys(mapObj);
	newfeatures = F(cell2mat(keySet),:); 		
end
	
function [newfeatures] = getNewfeatures(numOfsaes,hidden,D,threshold,F)
	disrec = getSymmatrix(numOfsaes*hidden,D);
	[B,I]=sort(D,'descend');					
	%max(B)
	%min(B)
	%mean(B)
	count = 1;
	f_num = 0;

	mapObj = containers.Map('KeyType','int32','ValueType','int32');

	while f_num <=hidden && count<=size(I,2)
		[r,c]=getIndices(I(count),numOfsaes,hidden);
		keySet = keys(mapObj);
		if isempty(keySet)
			mapObj(r)=r;
			mapObj(c)=c;
			count = count+1;
		else	
			if ~isKey(mapObj,r)&&~isKey(mapObj,c)
				addrto = true;			
				addcto = true;
			    for i = 1:size(keySet,2)
					%if ~((disrec(keySet{i},r)>=threshold)&&(disrec(keySet{i},c)>=threshold))
					%	addto = false;		
					%	break;
					%end
					if ~(disrec(keySet{i},r)>=threshold)
						addrto =false;	
					end

					if ~(disrec(keySet{i},c)>=threshold)
						addcto =false;	
					end	
				end

				if addrto
					mapObj(r)=r;
					f_num = f_num+1;
				end

				if addcto
					mapObj(c)=r;
					f_num = f_num+1;
				end
			end
			count=count+1;
		end
	end
	
	newfeatures = F(cell2mat(keySet),:); 		
end

function [filename]= getfilename(i)
	filename = strcat('opttheta',num2str(i),'.mat');
end

function [symmetric] = getSymmatrix(dim,data)
    M= triu(ones(dim),1)';
	M(~~M) = data;
	sym_indice = triu(true(size(M)),1)';		
	symmetric = M';
	symmetric(sym_indice)=data;
end

function [row,col]=getIndices(pos,numOfsaes,hidden)
	notFound = true;			
	len=numOfsaes*hidden-1;
	temp_len = 0;
	col = 1;
	while notFound&&len>0
		temp_len = temp_len+len;
		if pos>temp_len	
			len=len-1;
			col=col+1;
		else		
			row = numOfsaes*hidden-(temp_len-pos);
			notFound = false;
		end
	end
end
