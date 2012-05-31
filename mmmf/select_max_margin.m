function [i, j, evals] = select_max_margin(Ytr, can_query, x, xu, xv)

[M, N] = size(can_query);
[I, J] = find(can_query);

margin = abs(x(can_query(:)));
evals = sparse(I, J, margin, M, N);

[~, idx] = max(margin);
i = I(idx);
j = J(idx);
end
