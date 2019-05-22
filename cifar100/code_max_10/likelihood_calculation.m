function probability = likelihood_calculation(net, imdb, new_labels)

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
result = softmax_likelihood(squeeze(res(end).x)); 
probability(:,i) = result;

end




end
