% testing the tree

% generate the tree

%testing tree current road block is saving and loading nodes

load '../dataset/mnist_imdb.mat';
%load '../dataset/imdb_0to7.mat';

top_node = node;

top_node.id = 1;

netPath = '../output/tree_1/';

recursive_add_children(top_node,netPath)

% how to do batch processing ??? 

im = mnist_imdb.images.data(:,:,:,mnist_imdb.images.set==1);

labels = mnist_imdb.images.labels(1,mnist_imdb.images.set==1);

prediction = zeros(1,numel(labels));

% command to gene

% note to self: learn to execute by recursion
% generate tree before hand

for i = 1 : numel(labels)
    net = top_node.net;
    net.layers(end) = [];
    res = vl_simplenn(net,im(:,:,:,i));
    x = squeeze(res(end).x);
    [~,index] = max(x);
    
    next_node = top_node.children{index};
    net = next_node.net;
    net.layers(end) = [];
    res = vl_simplenn(net,im(:,:,:,i));
    x = squeeze(res(end).x);
    [~,index] = max(x);
    prediction(1,i) = next_node.labelsTransform(next_node.labelsTransform(:,2)==index,1);
end

error = ~bsxfun(@eq, prediction, labels) ;

fprintf('error: %.4f \n', sum(error)/numel(error));



