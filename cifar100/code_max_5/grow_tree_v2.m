function [labelsTransform, merge_info] = grow_tree_v2(value, index, labelsTransform, new_labels)
k = 1;
i_old = zeros(numel(new_labels),1);
lt_old = labelsTransform;
merge_info = [];
max_children = 5;
%first assign high values
while(~isempty(new_labels))
    
    [~,i] = max(value(1,:));
    v_col = value(:,i);
    i_col = index(:,i);
    start = 1;
    for j = 1:numel(i_col)
        if numel(labelsTransform(labelsTransform(:,2)==i_col(j)))< max_children
            start = j;
            break;
        end
    end
    
    if (v_col(start)-v_col(start+1) > 0.1)||(i_col(start) == i_col(start+1))
         labelsTransform(end+1,:) = [new_labels(i), i_col(start)];
        value(:,i) = [];
        index(:,i) = [];
        new_labels(i) = [];
        i_old(k) = i_col(start);  
    elseif (v_col(start+1) - v_col(start+2) > 0.1)
        % merge condition check 
        num_orig_children_2 = numel(lt_old(lt_old(:,2) == i_col(start+1),2));
        num_children_1 = numel(labelsTransform(labelsTransform(:,2) == i_col(start),2));
        num_children_2 = numel(labelsTransform(labelsTransform(:,2) == i_col(start+1),2));
        if num_orig_children_2 > 1
            merge = false;
        else
            total_children_after_merge = num_children_1 + num_children_2 + 1;
            if total_children_after_merge > max_children
                merge = false;
            else 
                merge = true;
            end
        end
        
        if merge
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
        else  %no merge add node to one with less children
            if num_children_1 <= num_children_2
                labelsTransform(end+1,:) = [new_labels(i), i_col(start)];
                value(:,i) = [];
                index(:,i) = [];
                new_labels(i) = [];
                i_old(k) = i_col(start); 
            else
                labelsTransform(end+1,:) = [new_labels(i), i_col(start+1)];
                value(:,i) = [];
                index(:,i) = [];
                new_labels(i) = [];
                i_old(k) = i_col(start+1); 
            end
        end
                
            
    else % Create new node
        n = max(labelsTransform(:,2));
        labelsTransform(end+1,:) = [new_labels(i), n+1];
        new_labels(i) = [];
        value(:,i) = [];
        index(:,i) = [];   
        i_old(k) = n+1;
    end      
    k= k+1;
end
% if ~(isempty(merge_info))
%     for l = 1:numel(merge_info(:,1))
%         for ll = 1:numel(labelsTransform(:,2))
%             if labelsTransform(ll,2) > merge_info(l,1)
%                 labelsTransform(ll,2) = labelsTransform(ll,2) - 1;
%             end
%             
%         end
%     end
% end
    

end

