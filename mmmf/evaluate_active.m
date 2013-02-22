function [all_results] = evaluate_active(...
    Y, selectors, steps, known, queryable, C, test_on, outfile)
% INPUTS:
%   Y: a -1/0/1 label matrix, where 0 means unknown
%   selectors: a cell array of function handles to use for prediciton
%   steps: the number of steps of prediction to run (-1 to go until end)
%   known: the initially known elements of the matrix
%   queryable: the elements that are allowed to be queried (default: Y ~= 0)
%   C: slack variable in MMMF (default 1 XXX)
%   test_on: the elements to test on (default: Y ~= 0 & ~known)
%
% OUTPUTS:
%   all_results: a cell array with one entry per selector, containing
%       cell arrays of num_known, misclassification rate, pick [i,j],
%                      evaluations array, predictions

if ~iscell(selectors); selectors = {selectors}; end
if nargin < 3; steps = -1; end
if nargin < 4; known = eye(size(Y)); end
if nargin < 5; queryable = (Y ~= 0); end
if nargin < 6; C = 1; end
% test_on handled later
if nargin < 8; save_partial = false; else; save_partial = true; end

addpath(genpath('yalmip'))
addpath('~/share/sedumi')

known = logical(known);
[known_i, known_j] = find(known);
num_known_init = nnz(known);

queryable = logical(queryable);
queryable(known) = false;

Ytr_init = double(zeros(size(Y)));
Ytr_init(known) = Y(known);

if nargin < 7 || numel(test_on) < 1
    test_on = (Y ~= 0) & (~known);
else
    test_on = logical(test_on);
end

function [err] = get_misclass(X)
    err = mean(Y(test_on) ~= sign(X(test_on)))
end

% initial fit
[x_init, xu_init, xv_init] = solveD(Ytr_init, 'a', C, 'sedumi');

all_results = cell(1, length(selectors));

for selector_i = 1 : length(selectors)
    selector = selectors{selector_i};
    Ytr = Ytr_init;
    x = x_init;
    xu = xu_init;
    xv = xv_init;
    num_known = num_known_init;
    can_query = queryable;

    results = cell(1, 5);
    results(1,:) = {num_known, get_misclass(x), [], [], x};

    stepnum = 2;
    while (steps == -1 || stepnum <= steps) && nnz(can_query) > 0
        % pick a query item
        if nnz(can_query) == 1
            [i, j] = find(can_query);
            evals = [];
        else
            [i, j, evals] = selector(Ytr, can_query, x, xu, xv);
        end

        % learn the value of that query item
        Ytr(i, j) = Y(i, j);
        can_query(i, j) = 0;
        [x, xu, xv] = solveD(Ytr, 'a', C, 'sedumi');
        num_known = num_known + 1;

        % save results
        results(stepnum, :) = {num_known, get_misclass(x), [i,j], evals, x};
        stepnum = stepnum + 1;
        if save_partial & mod(stepnum, 20) == 0
            save(outfile, 'results');
        end
    end

    all_results{selector_i} = results;
end
end
