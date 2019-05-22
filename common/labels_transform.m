function labels_new = labels_transform(labels, old, new)
n = numel(labels);
labels_new = zeros(size(labels));
for i = 1:n
    [check,loc] = ismember(labels(1,i),old);
    if (check)
        labels_new(1,i) = new(loc);
    else
        labels_new(1,i) = 101;
    end   
end


end