function [B,I] = fisherscore(weights,layerinput,inputlabels)
    layerinput = sigmoid(layerinput);
    weightsum = weights*layerinput;%200*183
    autism = weightsum(:,find(inputlabels==1));
    tdc = weightsum(:,find(inputlabels==2));
    autism_mean = mean(autism,2); % 1*200
    tdc_mean = mean(tdc,2);% 1*200
    autism_std= std(autism,1,2);  
    tdc_std= std(tdc,1,2);  
    fisher =(autism_mean-tdc_mean).^2./autism_std.^2-tdc_std.^2;
    [B,I]=sort(abs(fisher),'descend');
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
