function [i, j, evals] = select_random(~, mask, ~, ~, ~, ~, ~, ~)

[M, N] = size(mask);
[I, J] = find(mask);

criteria = rand(1, length(I)) + 1;
evals = sparse(I, J, criteria, M, N);
[~, idx] = max(criteria(:));
i = I(idx);
j = J(idx);

end
