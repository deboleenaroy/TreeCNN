% making change to a new branch
clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

load '../dataset/cifar_100_imdb.mat';

% 0 apple
% 1 aquarium fish 
% 2 baby
% 3 bear
% 5 bed
% 7 beetle
% 9 bottle
% 13 bus
% 23 cloud
% 29 dinosaur


reduced_labels = [0,1,2,3,5,7,9,13,23,29];

labelsTransform = [ 0,1; ...
    1,2; ...
    2,3; ...
    3,4; ...
    5,5; ...
    7,6; ...
    9,7; ...
    13,8; ...
    23,9; ...
    29,10; ];

imdb = create_reduced_imdb(cifar_100_imdb, labelsTransform);

save('../dataset/imdb_v1.mat','imdb');

load '../dataset/imdb_v1.mat';

node1 = node;

node1.net = init_dcnn();

node1.opts.gpus = [1];
node1.opts.continue = false;
node1.opts.plotStatistics = true;
node1.opts.batchSize = 25;
node1.opts.numEpochs = node1.net.meta.trainOpts.numEpochs;

node1.net.meta.classes.name = imdb.meta.classes(:)' ;

node_1.net.meta.labelstransform = labelsTransform;




node1.opts.expDir = '/data/roy77/DNN/T_CNN/cifar100/output/trial_2/';

[node1.net,info] = cnn_my_train(node1.net,imdb, node1.opts);










