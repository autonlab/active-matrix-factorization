function [i, j, evals] = select_1step_lowest_entropy(...
    Xtr, mask, P, ~, vals, featureFunc, lagrange, delta)
% Do one-step lookahead and select the point that results in the
% lowest maxent solution entropy.
%
% TODO: support not evaluating every possibility for every query point

[M, N] = size(mask);
[I, J] = find(mask);

expectations = nan(1, length(I));

for idx = 1 : length(I)
    exp = 0;
    
    this_i = I(idx);
    this_j = J(idx);
    
    new_mask = mask;
    new_mask(this_i, this_j) = 0;
    
    new_Xtr = Xtr;
    
    for val_i = 1 : length(vals)        
        new_Xtr(this_i, this_j) = vals(val_i);
        [~, new_P] = ratingconcentration(...
            new_Xtr, new_mask, featureFunc, delta, lagrange, vals);
        
        query_P = new_P(new_mask);
        entropy = -sum(query_P .* log(query_P));
        
        exp = exp + P(idx, val_i) * entropy;
    end
    
    expectations(idx) = exp;
end

evals = sparse(I, J, expectations, M, N);
[~, idx] = min(expectations(:));
i = I(idx);
j = J(idx);
end