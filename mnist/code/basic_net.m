% basic network 
clear;
setup_2;
addpath(genpath('../../common'));

basic = node;

basic.net = gen_model_A();
N = 50;

basic.net.meta.inputSize = [28 28 1] ;
basic.net.meta.trainOpts.learningRate = 0.5*ones(1,N) ;
basic.net.meta.trainOpts.weightDecay = 0 ;
basic.net.meta.trainOpts.batchSize =  N ;
basic.net.meta.trainOpts.numEpochs = numel(basic.net.meta.trainOpts.learningRate) ;

load '../dataset/imdb_0to7.mat';
basic.filePath = '../output/basic/initial/';
basic.opts = init_opts();
basic.opts.numEpochs = 50;
basic.opts.learningRate = 0.5*ones(1,50);
basic.opts.weightDecy = 0;
basic.opts.expDir = basic.filePath;
basic.net.meta.trainOpts.numEpochs = basic.opts.numEpochs;
basic.opts.continue = true;

basic.opts.plotStatistics = true;
basic.opts.learningRate = basic.net.meta.trainOpts.learningRate;

[basic.net,~]=cnn_my_train(basic.net,imdb_0to7,basic.opts);

copyfile(fullfile(basic.filePath,sprintf('net-epoch-%d.mat',basic.opts.numEpochs)), ...
    fullfile(basic.filePath,'net.mat'));







%--------- incremental learning

% by adding two new nodes for 8 and 9 

% varied over backprofdepth


clear imdb;
clear imdb_0to7;

load '../dataset/mnist_imdb.mat';

basic.net = loadnet(fullfile(basic.filePath, 'net.mat'));

%append new nodes

new = 2;

sz = size(basic.net.layers{end-1}.weights{1});

basic.net.layers{end-1}.weights{1} = 0.05*randn(sz(1),sz(2),sz(3),sz(4)+new, 'single');
basic.net.layers{7}.weights{2} = zeros(1,sz(4)+new, 'single');
opts = basic.opts;
backPropDepth = [2;5;7];

for i = 1 : numel(backPropDepth)
   net =  basic.net;
   opts.backPropDepth = backPropDepth(i);
   opts.expDir = fullfile(sprintf('../output/basic/incremental/backPropDepth-%d/', ...
                                   backPropDepth(i)));
   [net,~] = cnn_my_train(net,mnist_imdb,opts); 
   
    
end










