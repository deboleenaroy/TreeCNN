clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

% initialize tree

oldPath = '../output/tree_5/';
newPath = '../output/tree_6/' ;
if ~exist(newPath, 'dir')
    mkdir(newPath);
end

top_node_old = node;

top_node_old.filePath = fullfile(oldPath, sprintf('%d',top_node_old.id));

recursive_add_children(top_node_old, oldPath);

% 44 lizard
% 68 road
% 63 porcupine
% 50 mouse
% 72 seal
% 71 sea
% 88 tiger
% 86 telephone
% 69 rocket
% 92 tulip


load '../dataset/cifar_100_imdb.mat';

new_labels = [44;68;63;50;72;71;88;86;69;92];

net = top_node_old.net;
labelsTransform = top_node_old.labelsTransform;
 
%likelihood matrix calculation
 l_matrix = likelihood_calculation_limited(net,cifar_100_imdb,new_labels,labelsTransform);
 [value, index] = sort(l_matrix,'descend');
 save(fullfile(newPath, 'l_matrix.mat'), 'l_matrix','value','index');

load(fullfile(newPath, 'l_matrix.mat'));

[labelsTransform, ~] = grow_tree_2(value, index, labelsTransform, new_labels);



imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb, labelsTransform);
save('../dataset/imdb_v2.mat', 'imdb');

%load '../dataset/imdb_v2.mat';
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

top_node.opts.gpus = [3];
top_node.opts.continue = true;
top_node.opts.plotStatistics = true;
top_node.opts.batchSize = 50;
top_node.opts.numEpochs = 250;
top_node.opts.learningRate = [0.1*ones(1,50),0.01*ones(1,100), 0.001*ones(1,100)];
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

recursive_add_children(top_node, newPath);

for i = 1 : N
    next_node = top_node.children{i};
    
    
    labels = top_node.labelsTransform(top_node.labelsTransform(:,2)== i,1);
    
    if ~exist(next_node.filePath, 'dir')
        mkdir (next_node.filePath)
    end
    
    if numel(labels) == 1
        continue;
    else
        if numel(top_node_old.children) >= i
            next_node_old = top_node_old.children{i};
            labels_old = top_node_old.labelsTransform(top_node_old.labelsTransform(:,2)== i,1);
            if numel(labels) == numel(labels_old)
                next_node.net = next_node_old.net;
                copyfile(fullfile(next_node_old.filePath,'net.mat'), ...
                    fullfile(next_node.filePath,'net.mat'));
                copyfile(fullfile(next_node_old.filePath,'labelsTransform.mat'), ...
                    fullfile(next_node.filePath,'labelsTransform.mat'));
                continue;
            else
                next_node.labelsTransform = next_node_old.labelsTransform;
                net = next_node_old.net;
                if isempty(next_node_old.labelsTransform)
                    next_node.labelsTransform(:,1) = labels;
                    next_node.labelsTransform(:,2) = [1:1:numel(labels)];
                else
                    for j = 1:numel(labels)
                        if ~ismember(labels(j), next_node.labelsTransform(:,1))
                            n = numel(next_node.labelsTransform(:,1));
                            next_node.labelsTransform(end+1,:) = [labels(j), n+1];
                        end
                    end
                end
            end
        else
            next_node.labelsTransform(:,1) = labels;
            next_node.labelsTransform(:,2) = [1:1:numel(labels)];
        end
               
        
        imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb, next_node.labelsTransform);
        next_node.net = gen_model_B2(numel(unique(next_node.labelsTransform(:,2))));
        if ~isempty(next_node_old.net)
            for jj = 1:numel(next_node.net.layers)-8
                next_node.net.layers{jj} = next_node_old.net.layers{jj};
            end
        end
        
        next_node.opts.backPropDepth = +inf;
        next_node.opts.gpus = [2];
        next_node.opts.continue = true;
        next_node.opts.plotStatistics =true;
        next_node.opts.batchSize = 20;
        next_node.opts.numEpochs = 250;
        next_node.opts.learningRate = [0.1*ones(1,50), 0.01*ones(1,100), 0.001*ones(1,100)];
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
    