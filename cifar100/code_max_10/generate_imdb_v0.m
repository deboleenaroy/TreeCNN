function [labelsTransform, imdb] = generate_imdb_v0()

% 20 Chair
% 12 Bridge
% 35 Girl
% 38 kangaroo
% 41 lawn_mower
% 64 possum
% 55 otter
% 62 poppy
% 83 sweet_pepper
% 08 bicycle


load '../dataset/cifar_100_imdb.mat';
labelsTransform(:,1) = [20;12;35;38;41;64;55;62;83;8];
labelsTransform(:,2) = [1;2;3;4;5;6;7;8;9;10];

    

if exist('../dataset/imdb_v0.mat','file')
   load '../dataset/imdb_v0.mat'
else
    imdb = create_reduced_imdb_for_cifar100(cifar_100_imdb, labelsTransform);
    save('../dataset/imdb_v0.mat','imdb')
end


end
