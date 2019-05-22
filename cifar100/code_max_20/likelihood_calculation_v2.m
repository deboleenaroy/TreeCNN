function probability = likelihood_calculation_v2(net, imdb, new_labels, labelsTransform)

net.layers(end) = [];
limit_index = [];
k = 1;

for j = 1:numel(unique(labelsTransform(:,2)))
    if numel(labelsTransform(labelsTransform(:,2)==j,2)) == 20
        limit_index(k) = j;
        k = k+1;
    end    
end

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
    
    A = squeeze(res(end).x);
    p_sum = 0;
    l = 1;
    if ~isempty(limit_index)
        for l = 1:size(A, 1)
            if ~ismember(l, limit_index)
                p_sum = p_sum + exp(sum(A(l,:),2)/size(A(l,:),2));
            end
        end
        l = 1;
        for l = 1:size(A, 1)
            if ismember(l, limit_index)
                probability(l,i) = 0;
            else
                probability(l,i) = exp(sum(A(l,:),2)/size(A(l,:),2))/p_sum;
            end
        end
    else
        p_sum = exp(sum(A,2)/size(A,2));
        probability(:,i) = p_sum/sum(p_sum);
    end
    
end
end
