function probability = likelihood_calculation(net, imdb, new_labels)

net.layers(end) = [];

probability = single(zeros(size(net.layers{end}.weights{1},4),numel(new_labels)));

for i = 1:numel(new_labels)

im = imdb.images.data(:,:,:,imdb.images.labels == new_labels(i));

res = vl_simplenn(net,im(:,:,:,1:100));

% two available likelihood function
% max_likelihood : acts like a confusion matrix
% softmax_likelihood: acts like a softmax function

result = softmax_likelihood(squeeze(res(end).x)); 

probability(:,i) = result;

end




end
