% Take the old net, use top most convolutional block 

%first retrain top most block

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

% 11 boy
% 35 girl
% 53 orange
% 57 pear

clear;
setup;



% load '../dataset/cifar_100_imdb.mat';
% 
% reduced_labels = [0,1,2,3,5,7,9,13,23,29];
% 
% labelsTransform = [ 0,1; ...
%     1,2; ...
%     2,3; ...
%     3,4; ...
%     5,5; ...
%     7,6; ...
%     9,7; ...
%     13,8; ...
%     23,9; ...
%     29,10; ];
% 
% imdb = create_reduced_imdb(cifar_100_imdb,reduced_labels, labelsTransform);
% 
% imdb.meta.classes = {'apple'; 'fish'; 'baby'; 'bear'; 'bed'; 'beetle'; 'bottle'; 'bus'; 'cloud'; 'dinosaur'};
% 
% save('../dataset/imdb_1.mat','imdb');

load('../dataset/imdb_v1.mat');

netPath = '../output/Tree/10/net.mat';



node0 = node;

node0.net = loadnet(netPath);

node0.net.layers(end) = [];

node0.opts.test = find(imdb.images.set==1);
node0.opts.batchSize = 100;

[node0.net,error] = cnn_my_test(node0.net,imdb,node0.opts);




% create new blocks

