function [all_results] = evaluate_active(X, known, selectors, steps, delta, vals)

if nargin < 6; vals = unique(X); end
if nargin < 5; delta = 1.5; end % TODO: delta through CV
if nargin < 4; steps = -1; end
if ~iscell(selectors)
    selectors = {selectors};
end

if length(vals) == 5 && all(vals(:) == (1:5)')
    featureFunc = @sets_square5;
elseif length(vals) == 2 && all(vals(:) == (1:2)')
    featureFunc = @sets_onetwo;
else
    error('ratingconcentration:evaluate_active:vals', ...
          'Not sure how to do features for these values.')
end

featureFunc = @sets_square5;

mask_init = sparse(known == 0); % TODO: query on less than the full matrix

[known_i, known_j] = find(known);
num_known_init = nnz(known);

Xtr = sparse(known_i, known_j, X(known ~= 0));

% initial fit
[E, P, vals, lagrange] = ...
    ratingconcentration(Xtr, mask_init, featureFunc, delta, [], vals);
P = bsxfun(@rdivide, P, sum(P, 2)); % normalize prediction dists

all_results = cell(1, length(selectors));

for selector_i = 1 : length(selectors)
    selector = selectors{selector_i};
    num_known = num_known_init;
    mask = mask_init;

    results = cell(1, 4);
    results(1,:) = {num_known, get_rmse(X, E), [], []};

    stepnum = 2;
    while (steps == -1 || stepnum <= steps) && nnz(mask) > 0
        % pick a query item
        if nnz(mask) == 1
            [i, j] = find(mask);
            evals = [];
        else
            [i, j, evals] = selector(...
                Xtr, mask, P, E, vals, featureFunc, lagrange, delta);
        end

        % learn the value of that query item
        Xtr(i, j) = X(i, j); %#ok<SPRIX>
        mask(i, j) = 0; %#ok<SPRIX>
        [E, P, vals, lagrange] = ...
            ratingconcentration(Xtr, mask, @sets_square5, delta, [], vals);
        P = bsxfun(@rdivide, P, sum(P, 2));
        num_known = num_known + 1;

        % save results
        results(stepnum,:) = {num_known, get_rmse(X, E), [i,j], evals};
        stepnum = stepnum + 1;
    end

    all_results{selector_i} = results;
end
end

function [rmse] = get_rmse(X, E)
    rmse = sqrt(sum((X(:) - E(:)).^2) / numel(X));
end
