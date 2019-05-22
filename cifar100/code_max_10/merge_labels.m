function v1 = merge_labels(v1,l1,l2)
    for i = 1 : numel(v1)
        if v1(i) == l2
            v1(i) = l1;
        elseif v1(i) > l2
            v1(i) = v1(i) - 1;
        end
    end
end