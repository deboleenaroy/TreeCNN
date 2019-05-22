function [images, labels] = getBatch(imdb, batch, opts)
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.flip
    if rand > 0.5, images=fliplr(images) ; end
end
end