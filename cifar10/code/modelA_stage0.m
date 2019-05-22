clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));


filePath = '../output/basic_0/';

load '../dataset/cifar_10_imdb.mat';

labelsTransform(:,1) = [2;4;6;8;9;10];
labelsTransform(:,2) = [1:1:numel(labelsTransform(:,1))];

imdb = create_reduced_imdb_for_cifar10(cifar_10_imdb, labelsTransform);

net = gen_model_A(numel(unique(labelsTransform(:,2))));
opts = init_opts();
opts.gpus = [2];
opts.continue = true;
opts.plotStatistics = true;
opts.batchSize = 50;
opts.numEpochs = 350;
opts.learningRate = [0.1*ones(1,200), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];
opts.expDir = filePath;

net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.numEpochs = opts.numEpochs;
net.meta.trainOpts.weightDecay = opts.weightDecay ;
net.meta.trainOpts.batchSize = opts.batchSize ;
net.meta.classes.name = imdb.meta.classes(:)' ;

fprintf('starting training .. \n');

[net,~] = cnn_my_train(net,imdb, opts);

fprintf('saving files .. \n');

copyfile(fullfile(filePath,sprintf('net-epoch-%d.mat',opts.numEpochs)), ...
    fullfile(filePath,'net.mat'));

save(fullfile(filePath,'labelsTransform.mat'),'labelsTransform');

clear net;
clear labelsTransform;
clear imdb;






