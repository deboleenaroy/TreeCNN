function labelsTransform = grow_tree(value, index, labelsTransform, new_labels)
k = 1;
i_old = zeros(numel(new_labels),1);
%first assign high values
while(~isempty(new_labels))
    
    [v,i] = max(value(1,:));
    i_old(k) = index(1,i);
    
    
    if v > 0.55
        labelsTransform(end+1,:) = [new_labels(i), index(1,i)];
        value(:,i) = [];
        index(:,i) = [];
        new_labels(i) = [];
    else
        v2 = value(2,i);
        if((v-v2 > 0.1) || (index(2,i) == index(1,i)))
            labelsTransform(end+1,:) = [new_labels(i), index(1,i)];
            value(:,i) = [];
            index(:,i) = [];
            new_labels(i) = [];
        else 
            v3 = value(3,i);
            if (v-v3 > 0.1)
                if (v-v2 < 0.05) % Go ahead merge
                    [j,l] = min([index(1,i), index(2,i)]);
                    l2 = mod(l,2)+1;
                    labelsTransform(end+1,:) = [new_labels(i), j];
                    labelsTransform(:,2) = merge_labels(labelsTransform(:,2),j,index(l2,i));
                    value(:,i) = [];
                    index(:,i) = [];
                    i_old(k) = j;
                    new_labels(i) = [];
                    for col = 1: size(index,2)
                        index(:,col) = merge_labels(index(:,col),j, index(l2,i));
                    end
                else                
                    if (ismember(index(2,i),i_old))
                        labelsTransform(end+1,:) = [new_labels(i), index(1,i)];
                        value(:,i) = [];
                        index(:,i) = [];
                        new_labels(i) = [];
                    else
                        [j,l] = min([index(1,i), index(2,i)]);
                        l2 = mod(l,2)+1;
                        j2 = index(l2,i);
                        labelsTransform(end+1,:) = [new_labels(i), j];
                        labelsTransform(:,2) = merge_labels(labelsTransform(:,2),j,j2);                        
                        value(:,i) = [];
                        index(:,i) = [];
                        i_old(k) = j;
                        new_labels(i) = [];
                        for col = 1: size(index,2)
                            index(:,col) = merge_labels(index(:,col),j, j2);
                        end
                        
                    end
                end
            else
                i_old(k) = numel(unique(labelsTransform(:,2)))+1;
                labelsTransform(end+1,:) = [new_labels(i), i_old(k)];
                value(:,i) = [];
                index(:,i) = [];
                new_labels(i) = [];
            end
        end
    end
    k= k+1;
end 
end

