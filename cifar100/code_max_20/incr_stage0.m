clear;
setup;
addpath(genpath('/data/roy77/matconvnet'));
addpath(genpath('../../common'));

top_node = node;
top_node.filePath = fullfile(sprintf('../output_v2/tree_0/%d/',top_node.id));

[labelsTransform,imdb] = generate_imdb_v0();
fprintf('imdb load complete .. \n');

top_node.labelsTransform = labelsTransform;
top_node.net = gen_model_B(numel(unique(top_node.labelsTransform(:,2))));
top_node.opts.gpus = [1];
top_node.opts.continue = true;
top_node.opts.plotStatistics = true;
top_node.opts.batchSize = 25;
top_node.opts.numEpochs = 300;
top_node.opts.learningRate = [0.1*ones(1,100), 0.01*ones(1,100), 0.001*ones(1,50), 0.0001*ones(1,50)];
top_node.opts.expDir = top_node.filePath;

top_node.net.meta.trainOpts.learningRate = top_node.opts.learningRate;
top_node.net.meta.trainOpts.numEpochs = top_node.opts.numEpochs;
top_node.net.meta.trainOpts.weightDecay = top_node.opts.weightDecay ;
top_node.net.meta.trainOpts.batchSize = top_node.opts.batchSize ;
top_node.net.meta.classes.name = imdb.meta.classes(:)' ;

fprintf('starting training .. \n');

[top_node.net,~] = cnn_my_train(top_node.net,imdb, top_node.opts);

fprintf('saving files .. \n');

save(fullfile(top_node.filePath,'labelsTransform.mat'),'labelsTransform');

clear net;
clear labelsTransform;
clear imdb;






