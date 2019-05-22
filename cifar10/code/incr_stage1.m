clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

% first stage incremental learning 

% initialize tree

oldPath = '../output/tree_0/';
newPath = '../output/tree_1/' ;

top_node_old = node;

top_node_old.filePath = fullfile(oldPath, sprintf('%d',top_node_old.id));

recursive_add_children(top_node_old, oldPath);

load '../dataset/cifar_10_imdb.mat';

new_labels = [1;3;5;7];
net = top_node_old.net;
labelsTransform = top_node_old.labelsTransform;
l_matrix = likelihood_calculation(net,cifar_10_imdb,new_labels);
[value, index] = sort(l_matrix,'descend');
save(fullfile(newPath, 'l_matrix.mat'), 'l_matrix','value','index');
load(fullfile(newPath, 'l_matrix.mat'));
labelsTransform = grow_tree(value, index, labelsTransform, new_labels);

%--------------------------------
% generate new imdb for training 
%---------------------------------

imdb = create_reduced_imdb_for_cifar10(cifar_10_imdb, labelsTransform);
save('../dataset/imdb_v1.mat', 'imdb');

load '../dataset/imdb_v1.mat';

%defining new top_node
N = numel(unique(labelsTransform(:,2)));
top_node = node;
top_node.filePath = fullfile(newPath, sprintf('%d',top_node.id));
save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');
top_node.labelsTransform = labelsTransform;

% transfer layers
if N == numel(unique(top_node_old.labelsTransform(:,2)))
   top_node.net = top_node_old.net;
else
    top_node.net = gen_model_B(N);
    for i = 1:numel(top_node_old.net.layers)-2
        top_node.net.layers{i} = top_node_old.net.layers{i};
    end
end

top_node.opts.gpus = [4];
top_node.opts.continue = true;
top_node.opts.plotStatistics = true;
top_node.opts.batchSize = 50;
top_node.opts.numEpochs = 250;
top_node.opts.learningRate = [0.1*ones(1,100), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];
top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
top_node.net.meta.classes.name = imdb.meta.classes(:)' ;
top_node.opts.expDir = top_node.filePath;
fprintf('starting training .. \n');
[top_node.net,~] = cnn_my_train(top_node.net,imdb, top_node.opts);
fprintf('saving files .. \n');
copyfile(fullfile(top_node.filePath,sprintf('net-epoch-%d.mat',top_node.opts.numEpochs)), ...
         fullfile(top_node.filePath,'net.mat'));
clear net;
clear labelsTransform;
clear imdb;



% -----------------------------
% adding children
% -----------------------------
recursive_add_children(top_node, newPath);

for i = 1 : N
    next_node = top_node.children{i};
    labels = top_node.labelsTransform(top_node.labelsTransform(:,2)== i,1);
    net_old = loadnet(fullfile(oldPath,sprintf('%d',next_node.id),'net.mat'));
    labelsTransform_old = loadlt(fullfile(oldPath,sprintf('%d',next_node.id),'labelsTransform.mat'));
    next_node.labelsTransform = labelsTransform_old;
    if numel(labels) > numel(next_node.labelsTransform(:,1))
        for j = 1:numel(labels)
            if ~ismember(labels(j), next_node.labelsTransform(:,1))
                n = numel(next_node.labelsTransform(:,1));
                next_node.labelsTransform(end+1,:) = [labels(j), n+1];
            end
        end
        imdb = create_reduced_imdb_for_cifar10(cifar_10_imdb, next_node.labelsTransform);
        next_node.net = gen_model_B2(numel(unique(next_node.labelsTransform(:,2))));
        for jj = 1:numel(next_node.net.layers)-2
            next_node.net.layers{jj} = net_old.layers{jj};
        end
        
        next_node.opts.backPropDepth = +inf;
        next_node.opts.gpus = [4];
        next_node.opts.continue = true;
        next_node.opts.plotStatistics = true;
        next_node.opts.batchSize = 50;
        next_node.opts.numEpochs = 250;
        next_node.opts.learningRate = [0.1*ones(1,100), 0.01*ones(1,50), 0.001*ones(1,50), 0.001*ones(1,50)];
        next_node.net.meta.trainOpts.learningRate = next_node.opts.learningRate;
        next_node.net.meta.trainOpts.numEpochs = next_node.opts.numEpochs;
        next_node.net.meta.classes.name = imdb.meta.classes(:)' ;
        next_node.opts.expDir = next_node.filePath;
        fprintf('starting training .. \n');
        [next_node.net,~] = cnn_my_train(next_node.net,imdb, next_node.opts);
        fprintf('saving files .. \n');
        copyfile(fullfile(next_node.filePath,sprintf('net-epoch-%d.mat',next_node.opts.numEpochs)), ...
            fullfile(next_node.filePath,'net.mat'));
        labelsTransform = next_node.labelsTransform;
        save(fullfile(next_node.filePath,'labelsTransform.mat'),'labelsTransform');
        clear labelsTransform;
        clear imdb;
    end
    
 
end

