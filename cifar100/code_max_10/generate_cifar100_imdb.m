clear all;
load './dataset/cifar-100-matlab/meta.mat'

cifar_100_imdb.meta.sets = {'train','val','test'};

cifar_100_imdb.meta.coarse_classes = coarse_label_names;
cifar_100_imdb.meta.fine_classes = fine_label_names;


clear coarse_label_names;
clear fine_label_names;

load './dataset/cifar-100-matlab/train.mat';

im = permute(reshape(data',32,32,3,[]),[2 1 3 4]) ;

im = single(im);

imMean = mean(im(:,:,:,:), 4);
im = bsxfun(@minus, im,imMean);



opts.contrastNormalization = true;

opts.whitenData = true;

if opts.contrastNormalization
    z = reshape(im,[],50000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    im = reshape(z, 32, 32, 3, []) ;
end



if opts.whitenData
    z = reshape(im,[],50000) ;
    W = z(:,:)*z(:,:)'/50000 ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    im = reshape(z, 32, 32, 3, []) ;
end

cifar_100_imdb.images.data = im;

cifar_100_imdb.images.coarse_labels = coarse_labels';

cifar_100_imdb.images.fine_labels = fine_labels';
cifar_100_imdb.images.set = ones(1,numel(fine_labels));

clear coarse_labels;
clear fine_labels;
clear data;

load './dataset/cifar-100-matlab/test.mat';

im = permute(reshape(data',32,32,3,[]),[2 1 3 4]) ;

im = single(im);


im = bsxfun(@minus, im,imMean);



if opts.contrastNormalization
    z = reshape(im,[],10000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    im = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
    z = reshape(im,[],10000) ;
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    im = reshape(z, 32, 32, 3, []) ;
end

cifar_100_imdb.images.data = cat(4,cifar_100_imdb.images.data,im);

cifar_100_imdb.images.coarse_labels = cat(2,cifar_100_imdb.images.coarse_labels,coarse_labels');

cifar_100_imdb.images.fine_labels = cat(2,cifar_100_imdb.images.fine_labels,fine_labels');

cifar_100_imdb.images.set = cat(2,cifar_100_imdb.images.set, 3*ones(1,numel(fine_labels)));

save('./dataset/cifar_100_imdb.mat','cifar_100_imdb');