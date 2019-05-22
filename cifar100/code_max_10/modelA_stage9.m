clear;
setup;
addpath(genpath('/data/roy77/DNN/matconvnet'));
addpath(genpath('../../common'));

load '../dataset/cifar_100_imdb.mat';

oldPath = '../output/basic_8/';

labelsTransform(:,1) = [20;12;35;38;41;64;55;62;83;8;
                        43;46;56;85;96;10;49;36;21;23;
                        60;42;17;6;66;13;65;90;99;67;
                        84;1;25;18;95;82;91;14;74;37;
                        9;53;29;4;5;77;32;73;89;0;
                        98;34;45;75;16;93;24;30;3;58;
                        44;68;63;50;72;71;88;86;69;92;
                        2;48;31;22;47;51;57;54;79;52;
                        94;80;27;97;61;76;39;7;81;26;
                        78;40;15;59;19;11;70;33;87;28];
                    
labelsTransform(:,2) = [1:1:100]';

imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb,labelsTransform);

opts = init_opts();
opts.gpus = [2];
opts.continue = true;
opts.plotStatistics = true;
opts.batchSize = 50;
opts.numEpochs = 250;
opts.learningRate = [0.1*ones(1,150), 0.01*ones(1,100)];



backPropDepth = [8;16;24;32;40];

for i = 1 : numel(backPropDepth)
    net = loadnet(fullfile(sprintf('../output/basic_8/backPropDepth-%d/', ...
        backPropDepth(i)), 'net-epoch-250.mat'));
    new = 10;
    
    sz = size(net.layers{end-1}.weights{1});
    net.layers{end-1}.weights{1} = 0.05*randn(sz(1),sz(2),sz(3),sz(4)+new, 'single');
    net.layers{7}.weights{2} = zeros(1,sz(4)+new, 'single');
    net.meta.trainOpts.learningRate = opts.learningRate;
    net.meta.trainOpts.numEpochs = opts.numEpochs;
    net.meta.trainOpts.weightDecay = opts.weightDecay ;
    net.meta.trainOpts.batchSize = opts.batchSize ;
    net.meta.classes.name = imdb.meta.classes(:)' ;
    
    
    opts.backPropDepth = backPropDepth(i);
    opts.expDir = fullfile(sprintf('../output/basic_9/backPropDepth-%d/', ...
        backPropDepth(i)));
    [net,~] = cnn_my_train(net,imdb,opts); 
    
end
