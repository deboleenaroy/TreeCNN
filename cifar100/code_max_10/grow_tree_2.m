function [labelsTransform, merge_info] = grow_tree_2(value, index, labelsTransform, new_labels)
k = 1;
i_old = zeros(numel(new_labels),1);
lt_old = labelsTransform;
merge_info = [];
%first assign high values
while(~isempty(new_labels))
    
    [~,i] = max(value(1,:));
    v_col = value(:,i);
    i_col = index(:,i);
    start = 0;
    
    for j = 1:numel(i_col)
        if numel(labelsTransform(labelsTransform(:,2)==i_col(j)))<10
            start = j;
            break;
        end
    end
    
   
    if (v_col(start)-v_col(start+1) > 0.1)||(i_col(start+1) == i_col(start))
        labelsTransform(end+1,:) = [new_labels(i), i_col(start)];
        value(:,i) = [];
        index(:,i) = [];
        new_labels(i) = [];
        i_old(k) = i_col(start);  
    else
        if (v_col(start+1) - v_col(start+2) > 0.1)
            if numel(labelsTransform(labelsTransform(:,2) == i_col(start+1),2)) >= 5
                %no merge
                labelsTransform(end+1,:) = [new_labels(i), i_col(start)];
                value(:,i) = [];
                index(:,i) = [];
                new_labels(i) = [];
                i_old(k) = i_col(start);        
            elseif ((numel(lt_old(lt_old(:,2) == i_col(start),2)) == 1) || ...
                    (numel(lt_old(lt_old(:,2) == i_col(start+1),2)) == 1 )) && ...
                    (numel(labelsTransform(labelsTransform(:,2) == i_col(start),2)) ...
                    + numel(labelsTransform(labelsTransform(:,2) == i_col(start+1),2)) < 10)
                labelsTransform(end+1,:) = [new_labels(i), i_col(start)];
                for j = 1:numel(labelsTransform(:,2))
                    if labelsTransform(j,2) == i_col(start+1)
                        labelsTransform(j,2) = i_col(start);
                    end
                end
                merge_info(end+1,:) = [ i_col(start+1), i_col(start)];
                value(:,i) = [];
                index(:,i) = [];
                for row = 1:size(index,1)
                    for col = 1:size(index,2)
                        if index(row,col) == i_col(start+1)
                            index(row,col) = i_col(start);
                        end
                    end
                end
                new_labels(i) = [];
            end
        else
            n = max(labelsTransform(:,2));
            labelsTransform(end+1,:) = [new_labels(i), n+1];
            new_labels(i) = [];
            value(:,i) = [];
            index(:,i) = [];
        end           
    end      
    k= k+1;
end 
end

