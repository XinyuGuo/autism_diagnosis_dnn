function testreshape()
	m = [4,3;2,1];
	M = repmat(m,[3,1]);
	M
	MM = permute(reshape(M',[2,2,3]),[2,1,3]);	
	%MM = reshape(M',[2,2,3]);	
	size(MM)	
	for i = 1 : 3
		MM(i,:,:)
	end
end
