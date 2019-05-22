function probability = max_likelihood(A)

[~,index] = max(A);

[sz1,sz2] = size(A);

predicted_labels = zeros(sz1,sz2);

for i = 1:sz2
    predicted_labels(index(i),i) = 1;
end

p_sum = sum(predicted_labels,2);

probability = zeros(sz1,1);

for i = 1:numel(p_sum)
    
    probability(i) = p_sum(i)/sum(p_sum);
end



end