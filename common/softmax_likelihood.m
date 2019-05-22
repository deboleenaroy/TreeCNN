function probability = softmax_likelihood(A)

p_sum = exp(sum(A,2)/size(A,2));
probability = zeros(size(A,1),1);

for i = 1:numel(p_sum)
    
    probability(i) = p_sum(i)/sum(p_sum);
end



end