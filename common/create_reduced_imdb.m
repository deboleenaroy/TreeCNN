function imdb = create_reduced_imdb(imdb0,labelsTransform)

index = find(ismember(imdb0.images.fine_labels(1,:),labelsTransform(:,1)));

imdb.images.data = imdb0.images.data(:,:,:,index);

imdb.images.fine_labels = imdb0.images.fine_labels(:,index);

imdb.images.coarse_labels = imdb0.images.coarse_labels(:,index);

imdb.images.labels = labels_transform(imdb.images.fine_labels, labelsTransform(:,1), labelsTransform(:,2));

imdb.images.set = imdb0.images.set(:,index);

imdb.meta.sets = imdb0.meta.sets;

if isfield(imdb0.meta, 'orig')
imdb.meta.orig.fine_classes = imdb0.meta.orig.fine_classes;
else
imdb.meta.orig.fine_classes = imdb0.meta.fine_classes;

for i = 1:numel(fine_labels)
    imdb.meta.classes{i,1} = imdb.meta.orig.fine_classes(fine_labels(i)+1);
end


end
