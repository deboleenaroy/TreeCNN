% Author:
clear;
addpath(genpath('../dataset'));

% create reduced dataset for initial learning

load mnist_imdb;

index = find(mnist_imdb.images.labels == 1 | ...
             mnist_imdb.images.labels == 2 | ...
             mnist_imdb.images.labels == 3 | ...
             mnist_imdb.images.labels == 4 | ...
             mnist_imdb.images.labels == 5 | ...
             mnist_imdb.images.labels == 6 | ...
             mnist_imdb.images.labels == 7 | ...
             mnist_imdb.images.labels == 8 );

imdb_0to7.meta.sets = mnist_imdb.meta.sets;
imdb_0to7.meta.classes = {'0';'1'; '2'; '3'; '4'; '5'; '6'; '7'};
imdb_0to7.images.data = mnist_imdb.images.data(:,:,:,index);
imdb_0to7.images.labels = mnist_imdb.images.labels(:,index);
imdb_0to7.images.set = mnist_imdb.images.set(:,index);

save('../dataset/imdb_0to7.mat','imdb_0to7');
