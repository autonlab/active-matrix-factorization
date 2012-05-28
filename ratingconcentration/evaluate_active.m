function [all_results] = evaluate_active(X, known, selectors, steps, delta, vals, pred_mode)

if nargin < 7; pred_mode = false; end
if nargin < 6; vals = unique(X); end
if nargin < 5; delta = 1.5; end % TODO: delta through CV
if nargin < 4; steps = -1; end
if ~iscell(selectors)
    selectors = {selectors};
end

function [rmse] = get_rmse(E, P)
    if pred_mode
        [~, pred] = max(P, [], 2);
    else
        pred = E;
    end
    rmse = sqrt(sum((X(:) - pred(:)).^2) / numel(X));
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

Xtr_init = sparse(known_i, known_j, X(known ~= 0));

% initial fit
[E_init, P_init, vals, lagrange_init] = ...
    ratingconcentration(Xtr_init, mask_init, featureFunc, delta, [], vals);
P_init = bsxfun(@rdivide, P_init, sum(P_init, 2)); % normalize prediction dists

all_results = cell(1, length(selectors));

for selector_i = 1 : length(selectors)
    selector = selectors{selector_i};
    num_known = num_known_init;
    mask = mask_init;
    Xtr = Xtr_init;
    lagrange = lagrange_init;
    E = E_init;
    P = P_init;

    results = cell(1, 4);
    results(1,:) = {num_known, get_rmse(E, P), [], []};

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
        [E, P, ~, lagrange] = ...
           ratingconcentration(Xtr, mask, @sets_square5, delta, lagrange, vals);
        P = bsxfun(@rdivide, P, sum(P, 2));
        num_known = num_known + 1;

        % save results
        results(stepnum,:) = {num_known, get_rmse(E, P), [i,j], evals};
        stepnum = stepnum + 1;
    end

    all_results{selector_i} = results;
end
end
