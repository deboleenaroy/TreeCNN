function [labelsTransform, imdb] = generate_imdb_v0()
% 2 automobile
% 4 cat
% 6 dog
% 8 horse
% 9 ship
% 10 truck


load '../dataset/cifar_10_imdb.mat';
labelsTransform(:,1) = [2;4;6;8;9;10];
labelsTransform(:,2) = [1;2;2;2;1;1];
    

if exist('../dataset/imdb_v0.mat','file')
   load '../dataset/imdb_v0.mat'
else
    imdb = create_reduced_imdb_for_cifar10(cifar_10_imdb, labelsTransform);
    save('../dataset/imdb_v0.mat','imdb')
end


end