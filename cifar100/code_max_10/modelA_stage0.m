clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));


filePath = '../output/trial_04_05/';

[labelsTransform,imdb] = generate_imdb_v0();
fprintf('imdb load complete .. \n');


net = gen_model_A(numel(unique(labelsTransform(:,2))));
opts = init_opts();
opts.gpus = [3];
opts.continue = true;
opts.plotStatistics = true;
opts.batchSize = 20;
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






