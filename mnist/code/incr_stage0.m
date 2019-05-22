% code for initilization
% Tree form for MNIST
% For MNIST, we grouped the digits in 2 groups,
clear;
setup_2;
addpath(genpath('../../common'));

load '../dataset/imdb_0to7.mat';

top_node = node;

top_node.filePath = fullfile(sprintf('../output/tree_0/%d/',top_node.id));

if exist(fullfile(top_node.filePath,'net.mat'),'file')
    top_node.net = loadnet(fullfile(top_node.filePath,'net.mat'));
    top_node.labelsTransform = loadlt(fullfile(top_node.filePath,'labelsTransform.mat'));
else
    top_node.labelsTransform = [1,1; ...
        2,2; ...
        3,2; ...
        4,2; ...
        5,1; ...
        6,2; ...
        7,1; ...
        8,1;];
    imdb = imdb_0to7;
    imdb.images.labels = labels_transform(imdb_0to7.images.labels, ...
        top_node.labelsTransform(:,1), top_node.labelsTransform(:,2));
       
    imdb.meta.classes = {'A';'B'};
    top_node.opts.expDir = top_node.filePath;
    top_node.opts.numEpochs = 50;
    top_node.opts.learningRate = 0.5*ones(1,top_node.opts.numEpochs);
    
    top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
    top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
    
    top_node.opts.plotStatistics = true;
  
    top_node.net = gen_model_B(2);
    [top_node.net, ~] = cnn_my_train(top_node.net, imdb, top_node.opts);
    
    
    copyfile(fullfile(top_node.filePath,sprintf('net-epoch-%d.mat',top_node.opts.numEpochs)), ...
    fullfile(top_node.filePath,'net.mat'));

    labelsTransform = top_node.labelsTransform;
    save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');
    clear net;
    clear labelsTransform;
    clear imdb;
end

% adding node A and node B

node_A = node;
top_node.addNode(node_A);

node_A.filePath = fullfile(sprintf('../output/tree_0/%d/',node_A.id));

if exist(fullfile(node_A.filePath,'net.mat'),'file')
    node_A.net = loadnet(fullfile(node_A.filePath,'net.mat'));
    node_A.labelsTransform = loadlt(fullfile(node_A.filePath,'labelsTransform.mat'));
else
    node_A.labelsTransform(:,1) = top_node.labelsTransform(top_node.labelsTransform(:,2)==1,1);
    node_A.labelsTransform(:,2) = 1:1:size(node_A.labelsTransform,1);
    node_A.net = gen_model_B2(numel(unique(node_A.labelsTransform(:,2))));
    
    node_A.opts.expDir = node_A.filePath;
    node_A.opts.numEpochs = 50;
    node_A.opts.learningRate = 0.5*ones(1,top_node.opts.numEpochs);
    
    node_A.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
    node_A.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
    
    node_A.opts.plotStatistics = true;
    %transfer learning
%     node_A.net.layers{1} = top_node.net.layers{1};
%     node_A.opts.backPropDepth = 5;
    node_A.opts.backPropDepth = +inf;
    imdb = create_reduced_imdb_for_mnist(imdb_0to7, node_A.labelsTransform);
    imdb.meta.classes = {'0';'4';'6';'7'}; 
    
    [node_A.net, ~] = cnn_my_train(node_A.net, imdb, node_A.opts);
       
    copyfile(fullfile(node_A.filePath,sprintf('net-epoch-%d.mat',node_A.opts.numEpochs)), ...
    fullfile(node_A.filePath,'net.mat'));
    labelsTransform = node_A.labelsTransform;
    save(fullfile(node_A.filePath,'labelsTransform.mat'),'labelsTransform');
    
    clear net;
    clear labelsTransform;
    clear imdb;
end
    
    
node_B = node;
top_node.addNode(node_B);

node_B.filePath = fullfile(sprintf('../output/tree_0/%d/',node_B.id));

if exist(fullfile(node_B.filePath,'net.mat'),'file')
    node_B.net = loadnet(fullfile(node_B.filePath,'net.mat'));
    node_B.labelsTransform = loadlt(fullfile(node_B.filePath,'labelsTransform.mat'));
else
    node_B.labelsTransform(:,1) = top_node.labelsTransform(top_node.labelsTransform(:,2)==2,1);
    node_B.labelsTransform(:,2) = 1:1:size(node_B.labelsTransform,1);
    node_B.net = gen_model_B2(numel(unique(node_B.labelsTransform(:,2))));
    node_B.opts.numEpochs = 50;
    node_B.opts.learningRate = 0.5*ones(1,top_node.opts.numEpochs);
    
    node_B.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
    node_B.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
    node_B.opts.expDir = node_B.filePath;
%     %transfer learning
%     node_B.net.layers{1} = top_node.net.layers{1};
%     node_B.opts.backPropDepth = +inf;
%     node_B.net.layers{1}.learningRate = 0.1*node_B.net.layers{1}.learningRate;
    node_B.opts.backPropDepth = +inf;
    imdb = create_reduced_imdb_for_mnist(imdb_0to7, node_B.labelsTransform);
    imdb.meta.classes = {'1';'2'; '3'; '5'}; 
    
    [node_B.net, ~] = cnn_my_train(node_B.net, imdb, node_B.opts);
    copyfile(fullfile(node_A.filePath,sprintf('net-epoch-%d.mat',node_A.opts.numEpochs)), ...
    fullfile(node_A.filePath,'net.mat'));
    labelsTransform = node_B.labelsTransform;
    save(fullfile(node_B.filePath,'labelsTransform.mat'),'labelsTransform');
    
    clear net;
    clear labelsTransform;
    clear imdb;
end




