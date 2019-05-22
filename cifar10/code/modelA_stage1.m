clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

load '../dataset/cifar_10_imdb.mat';

oldPath = '../output/basic_0/';

net = loadnet(fullfile(oldPath,'net.mat'));

imdb = cifar_10_imdb;

new = 4;

sz = size(net.layers{end-1}.weights{1});

net.layers{end-1}.weights{1} = 0.05*randn(sz(1),sz(2),sz(3),sz(4)+new, 'single');
net.layers{7}.weights{2} = zeros(1,sz(4)+new, 'single');

opts = init_opts();
opts.gpus = [3];
opts.continue = true;
opts.plotStatistics = true;
opts.batchSize = 50;
opts.numEpochs = 250;
opts.learningRate = [0.1*ones(1,100), 0.01*ones(1,50), 0.001*ones(1,50), 0.0001*ones(1,50)];

net.meta.trainOpts.learningRate = opts.learningRate;
net.meta.trainOpts.numEpochs = opts.numEpochs;
net.meta.trainOpts.weightDecay = opts.weightDecay ;
net.meta.trainOpts.batchSize = opts.batchSize ;
net.meta.classes.name = imdb.meta.classes(:)' ;
backPropDepth = [8;16;24;32;40];

for i = 1 : numel(backPropDepth)
   net_temp =  net;
   opts.backPropDepth = backPropDepth(i);
   opts.expDir = fullfile(sprintf('../output/basic_1/backPropDepth-%d/', ...
                                   backPropDepth(i)));
   [net_temp,~] = cnn_my_train(net_temp,imdb,opts);   
end