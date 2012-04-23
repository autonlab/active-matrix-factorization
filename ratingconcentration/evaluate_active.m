function [ results ] = evaluate_active(X, known, selector, steps, delta, initial)

if nargin < 5; delta = 1.5; end % TODO: delta through CV
if nargin < 4; steps = -1; end

mask = sparse(known == 0); % TODO: query on less than the full matrix

[known_i, known_j] = find(known);
num_known = nnz(known);

Xtr = sparse(known_i, known_j, X(known));

if nargin < 6 || isempty(initial)
    % initial fit
    [E, P, vals, lagrange] = ...
        ratingconcentration(Xtr, mask, @sets_square5, delta);
    P = bsxfun(@rdivide, P, sum(P, 2)); % normalize prediction dists
else
    [E, P, vals, lagrange] = initial{:};
end

results = cell(1, 4);
results(1,:) = {num_known, get_rmse(X, E), [], []};

stepnum = 2;
while (steps == -1 || stepnum <= steps) && any(mask(:))
    % pick a query item
    if nnz(mask) == 1
        [i, j] = find(mask);
        evals = [];
    else
        [i, j, evals] = selector(...
            Xtr, mask, P, E, vals, featureFunc, lagrange, delta);
    end
    
    % learn the value of that query item
    Xtr(i, j) = X(i, j);
    mask(i, j) = 0;
    [E, P, vals, lagrange] = ...
        ratingconcentration(Xtr, mask, @sets_square5, delta);
    P = bsxfun(@rdivide, P, sum(P, 2));
    num_known = num_known + 1;
    
    % save results
    results(stepnum,:) = {num_known, get_rmse(X, E), [i,j], evals};
    stepnum = stepnum + 1;
end
end

function [rmse] = get_rmse(X, E)
    rmse = sqrt(sum((X(:) - E(:)).^2) / numel(X));
end