clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

newPath = '../output/tree_0/';
top_node = node;
top_node.filePath = fullfile(sprintf('../output/tree_0/%d/',top_node.id));

[labelsTransform,imdb] = generate_imdb_v0();
fprintf('imdb load complete .. \n');

top_node.labelsTransform = labelsTransform;
top_node.net = gen_model_B(numel(unique(top_node.labelsTransform(:,2))));
top_node.opts.gpus = [3];
top_node.opts.continue = true;
top_node.opts.plotStatistics = true;
top_node.opts.batchSize = 50;
top_node.opts.numEpochs = 350;
top_node.opts.learningRate = [0.1*ones(1,200), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];
top_node.opts.expDir = top_node.filePath;

top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
top_node.net.meta.trainOpts.weightDecay = top_node.opts.weightDecay ;
top_node.net.meta.trainOpts.batchSize = top_node.opts.batchSize ;
top_node.net.meta.classes.name = imdb.meta.classes(:)' ;

fprintf('starting training .. \n');

[top_node.net,~] = cnn_my_train(top_node.net,imdb, top_node.opts);

fprintf('saving files .. \n');

copyfile(fullfile(top_node.filePath,sprintf('net-epoch-%d.mat',top_node.opts.numEpochs)), ...
    fullfile(top_node.filePath,'net.mat'));

save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');

clear net;
clear labelsTransform;
clear imdb;

recursive_add_children(top_node, newPath);
load '../dataset/cifar_10_imdb.mat';
N = numel(unique(top_node.labelsTransform(:,2)));

for i = 1 : N
    
    if numel(find(top_node.labelsTransform(:,2) == i)) > 1
        next_node = top_node.children{i};  
        labels = top_node.labelsTransform(top_node.labelsTransform(:,2)== i,1);
        labelsTransform = [];
        labelsTransform(:,1) = labels;
        labelsTransform(:,2) = [1:1:numel(labels)]';
        next_node.labelsTransform = labelsTransform;
        next_node.labelsTransform = labelsTransform;
        imdb = create_reduced_imdb_for_cifar10(cifar_10_imdb, labelsTransform);
        next_node.net = gen_model_B2(numel(unique(next_node.labelsTransform(:,2))));
        next_node.opts.backPropDepth = +inf;
        next_node.opts.gpus = [3];
        next_node.opts.continue = true;
        next_node.opts.plotStatistics = true;
        next_node.opts.batchSize = 50;
        next_node.opts.numEpochs = 350;
        next_node.opts.learningRate = [0.1*ones(1,200), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];
        next_node.net.meta.trainOpts.learningRate = next_node.opts.learningRate;
        next_node.net.meta.trainOpts.numEpochs = next_node.opts.numEpochs;
        next_node.net.meta.classes.name = imdb.meta.classes(:)' ;
        next_node.opts.expDir = next_node.filePath;
        fprintf('starting training .. \n');
        [next_node.net,~] = cnn_my_train(next_node.net,imdb, next_node.opts);
        fprintf('saving files .. \n');
        copyfile(fullfile(next_node.filePath,sprintf('net-epoch-%d.mat',next_node.opts.numEpochs)), ...
        fullfile(next_node.filePath,'net.mat'));
        %labelsTransform = next_node.labelsTransform;
        %commenting because redundant
        save(fullfile(next_node.filePath,'labelsTransform.mat'),'labelsTransform');
    	clear labelsTransform;
        clear imdb;
    end
 
end
