clear;
modelPath = '../output/tree_0/';

top_node = node;

top_node.filePath = fullfile(modelPath, sprintf('%d',top_node.id));

recursive_add_children(top_node,modelPath)

load '../dataset/mnist_imdb.mat';

new_labels = [9;10];
N = 50;
net = top_node.net;

%likelihood matrix 

l_matrix = likelihood_calculation(net,mnist_imdb,new_labels);


[value, index] = max(l_matrix);

for i = 1 : numel(value)
    top_node.labelsTransform(end+1,:) = [new_labels(i),index(i)];
end

imdb = mnist_imdb;

imdb.images.labels = labels_transform(mnist_imdb.images.labels, ...
                      top_node.labelsTransform(:,1), top_node.labelsTransform(:,2));

imdb.meta.classes = {'A';'B'};

top_node.filePath = fullfile('../output/tree_1/', sprintf('%d',top_node.id));
top_node.opts.expDir = top_node.filePath;
top_node.opts.numEpochs = N;
top_node.opts.learningRate = 0.01*ones(1,top_node.opts.numEpochs);
top_node.opts.backPropDepth = +inf;
top_node.opts.continue = true;
top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
top_node.opts.plotStatistics = true;

[top_node.net, ~] = cnn_my_train(top_node.net, imdb, top_node.opts);

fprintf('saving files .. \n');
copyfile(fullfile(top_node.filePath,sprintf('net-epoch-%d.mat',top_node.opts.numEpochs)), ...
         fullfile(top_node.filePath,'net.mat'));
     
labelsTransform = top_node.labelsTransform;
%commenting because redundant
save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');


for i = 1 : numel(top_node.children)
    next_node = top_node.children{i};
    % we know this one only brnaches to final leaf nodes. so we add a leaf
    % node of the new class. 
    labels = top_node.labelsTransform(top_node.labelsTransform(:,2)==i,1);
    if numel(labels) > numel(next_node.labelsTransform(:,1))
        for j = 1:numel(labels)
            if ~ismember(labels(j), next_node.labelsTransform(:,1))
                n = numel(next_node.labelsTransform(:,1));
                next_node.labelsTransform(end+1,:) = [labels(j), n+1];
            end
        end
        imdb = create_reduced_imdb_for_mnist(mnist_imdb, next_node.labelsTransform);
        %imdb.meta.classes = 
        next_node.net = gen_model_B2(numel(unique(next_node.labelsTransform(:,2))));
        next_node.net.layers{1} = top_node.net.layers{1};
        next_node.opts.backPropDepth = 5;
        
        next_node.filePath = fullfile('../output/tree_1/', sprintf('%d',next_node.id));
        next_node.opts.expDir = next_node.filePath;
        next_node.opts.numEpochs = N;
        next_node.opts.learningRate = 0.01*ones(1,next_node.opts.numEpochs);
        next_node.opts.continue = true;
        next_node.net.meta.trainOpts.numEpochs = next_node.opts.numEpochs;
        next_node.net.meta.trainOpts.learningRate = next_node.opts.learningRate;
        next_node.opts.plotStatistics = true;
        
        [next_node.net,~] = cnn_my_train(next_node.net, imdb, next_node.opts);
        
        fprintf('saving files .. \n');
        copyfile(fullfile(next_node.filePath,sprintf('net-epoch-%d.mat',next_node.opts.numEpochs)), ...
            fullfile(next_node.filePath,'net.mat'));
        
        labelsTransform = next_node.labelsTransform;
        %commenting because redundant
        save(fullfile(next_node.filePath,'labelsTransform.mat'),'labelsTransform');


        
    end
end

