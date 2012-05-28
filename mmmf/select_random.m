function [i, j, evals] = select_random(~, can_query, ~, ~, ~)

[M, N] = size(can_query);
[I, J] = find(can_query);

criteria = rand(1, length(I)) + 1;
evals = sparse(I, J, criteria, M, N);
[~, idx] = max(criteria(:));
i = I(idx);
j = J(idx);
end
