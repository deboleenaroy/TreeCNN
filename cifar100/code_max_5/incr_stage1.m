clear;
setup;
addpath(genpath('/data/roy77/matconvnet'));
addpath(genpath('../../common'));

% first stage incremental learning 

% initialize tree

oldPath = './output/tree_0/';
newPath = './output/tree_1/' ;

if ~exist(newPath, 'dir')
    mkdir(newPath);
end

top_node_old = node;

top_node_old.filePath = fullfile(oldPath, sprintf('%d',top_node_old.id));

recursive_add_children_v2(top_node_old, oldPath);

% 43 lion
% 46 man
% 56 palm_tree
% 85 tank
% 96 willow_tree
% 10 bowl
% 49 mountain
% 36 hamster
% 21 chimpanzee
% 23 cloud

load '/data/roy77/Datasets/cifar_100_imdb.mat';

new_labels = [43;46;56;85;96;10;49;36;21;23];

net = top_node_old.net;
labelsTransform = top_node_old.labelsTransform;

%likelihood matrix 

l_matrix = likelihood_calculation_v2(net,cifar_100_imdb,  new_labels, labelsTransform);
[value, index] = sort(l_matrix,'descend');
save(fullfile(newPath, 'l_matrix.mat'), 'l_matrix','value','index');
%load './output/tree_1/l_matrix.mat';

labelsTransform = grow_tree_v2(value, index, labelsTransform, new_labels);


%--------------------------------
% generate new imdb for training 
%---------------------------------


imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb, labelsTransform);
% save('../dataset/imdb_v1.mat', 'imdb');
%load '../dataset/imdb_v1.mat';

%defining new top_node
N = numel(unique(labelsTransform(:,2)));
top_node = node;
top_node.filePath = fullfile(newPath, sprintf('%d',top_node.id));

top_node.labelsTransform = labelsTransform;
if ~exist(fullfile(newPath,sprintf('%d',top_node.id)),'dir')
    mkdir(fullfile(newPath,sprintf('%d',top_node.id)));
end
save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');
% transfer layers
if N == numel(unique(top_node_old.labelsTransform(:,2)))
   top_node.net = top_node_old.net;
else
    top_node.net = gen_model_B(N);
    for i = 1:numel(top_node_old.net.layers)-2
        top_node.net.layers{i} = top_node_old.net.layers{i};
    end
end

top_node.opts.gpus = [1];
top_node.opts.continue = false;
top_node.opts.plotStatistics = true;
top_node.opts.batchSize = 50;
top_node.opts.numEpochs = 100;
top_node.opts.learningRate = [0.1*ones(1,25), 0.01*ones(1,75), 0.001*ones(1,50), 0.0001*ones(1,50)];
top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
top_node.net.meta.classes.name = imdb.meta.classes(:)' ;
top_node.opts.expDir = top_node.filePath;
%fprintf('starting training .. \n');
[top_node.net,~] = cnn_my_train(top_node.net,imdb, top_node.opts);
fprintf('saving files .. \n');
clear net;
clear labelsTransform;
clear imdb;


% -----------------------------
% adding children
% -----------------------------
recursive_add_children_v2(top_node, newPath);



for i = 1 : N
    if numel(find(top_node.labelsTransform(:,2) == i)) > 1
        next_node = top_node.children{i};  
        labels = top_node.labelsTransform(top_node.labelsTransform(:,2)== i,1);
        labelsTransform = [];
        labelsTransform(:,1) = labels;
        labelsTransform(:,2) = [1:1:numel(labels)]';
        next_node.labelsTransform = labelsTransform;
        imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb, labelsTransform);
        next_node.net = gen_model_B2(numel(unique(next_node.labelsTransform(:,2))));
        next_node.opts.backPropDepth = +inf;
        next_node.opts.gpus = [1];
        next_node.opts.continue = false;
        next_node.opts.plotStatistics = true;
        next_node.opts.batchSize = 25;
        next_node.opts.numEpochs = 200;
        next_node.opts.learningRate = [0.1*ones(1,50), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];
        next_node.net.meta.trainOpts.learningRate = next_node.opts.learningRate;
        next_node.net.meta.trainOpts.numEpochs = next_node.opts.numEpochs;
        next_node.net.meta.classes.name = imdb.meta.classes(:)' ;
        next_node.opts.expDir = next_node.filePath;
        fprintf('starting training .. \n');
        [next_node.net,~] = cnn_my_train(next_node.net,imdb, next_node.opts);
        fprintf('saving files .. \n');
        save(fullfile(next_node.filePath,'labelsTransform.mat'),'labelsTransform');
    	clear labelsTransform;
        clear imdb;
    end
 
end






 
