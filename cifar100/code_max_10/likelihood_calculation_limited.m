function probability = likelihood_calculation_limited(net, imdb, new_labels, labelsTransform)

limit_index = [];
k = 1;

for i = 1:numel(unique(labelsTransform(:,2)))
    if numel(labelsTransform(labelsTransform(:,2)==i,2)) == 10
        limit_index(k) = i;
        k = k+1;
    end    
end


net.layers(end) = [];


probability = single(zeros(size(net.layers{end}.weights{1},4),numel(new_labels)));

for i = 1:numel(new_labels)
    
dzdy = [];
res = [];

im = imdb.images.data(:,:,:,imdb.images.fine_labels == new_labels(i));
res = vl_simplenn(net, im(:,:,:,1:50), dzdy, res, ...
                      'accumulate', false, ...
                      'mode', 'test', ...
                      'conserveMemory', true, ...
                      'backPropDepth', +inf, ...
                      'sync', true, ...
                      'cudnn', true, ...
                      'parameterServer', [], ...
                      'holdOn', false);
 
%result = softmax_likelihood(squeeze(res(end).x)); 
x = squeeze(res(end).x);

p_sum = exp(sum(x,2)/size(x,2));

for j = 1:numel(limit_index)
    p_sum(limit_index(j)) = 0;
end
for k = 1:numel(p_sum)
    
    probability(k,i) = p_sum(k)/sum(p_sum);
end



end




end