function [h] = select_ge_cutoff(cutoff)
    function [i, j, evals] = inner(~, mask, P, ~, vals, ~, ~, ~)
        [M, N] = size(mask);
        [I, J] = find(mask);

        probs = sum(P(mask(:), vals >= cutoff), 2);
        evals = sparse(I, J, probs, M, N);

        [~, idx] = max(probs);
        i = I(idx);
        j = J(idx);
    end

    h = @inner;
end
