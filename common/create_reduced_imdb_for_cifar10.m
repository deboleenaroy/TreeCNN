function imdb = create_reduced_imdb_for_cifar10(imdb0,labelsTransform)

index = find(ismember(imdb0.images.labels(1,:),labelsTransform(:,1)));

imdb.images.data = imdb0.images.data(:,:,:,index);

imdb.images.labels = imdb0.images.labels(:,index);

imdb.images.labels = labels_transform(imdb.images.labels, labelsTransform(:,1), labelsTransform(:,2));

imdb.images.set = imdb0.images.set(:,index);

imdb.meta.sets = imdb0.meta.sets;


for i = 1:numel(labelsTransform(:,1))
    imdb.meta.classes{i,1} = imdb0.meta.classes(labelsTransform(i,1));
end


end
