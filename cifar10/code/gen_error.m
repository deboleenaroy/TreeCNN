clear;
load '../output/basic_1/backPropDepth-40/net-epoch-250.mat';

error.val = zeros(25,1);
error.train = zeros(25,1);

for i = 10:10:250
    error.train(i/10) = stats.train(i).top1err;
    error.val(i/10) = stats.val(i).top1err;
end