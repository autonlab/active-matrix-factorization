function [h] = select_ge_cutoff(cutoff)
    function [i, j, evals] = inner(~, mask, P, ~, vals, ~, ~, ~)
        [M, N] = size(mask);
        [I, J] = find(mask);
        
        evals = sparse(I, J, sum(P(mask(:), vals >= cutoff), 2), M, N);
        
        [~, idx] = max(evals(:));
        [i, j] = ind2sub([M,N], idx);
    end

    h = @inner;
end