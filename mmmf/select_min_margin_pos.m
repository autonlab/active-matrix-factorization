function [i, j, evals] = select_min_margin_pos(Ytr, can_query, x, xu, xv)

[M, N] = size(can_query);
[I, J] = find(can_query);

margin = x(can_query(:));
margin(margin <= 0) = inf;
evals = sparse(I, J, margin, M, N);

[~, idx] = min(margin);
i = I(idx);
j = J(idx);
end
