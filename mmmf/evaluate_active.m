function [all_results] = evaluate_active(Y, selectors, steps, known, queryable, C)
% INPUTS:
%   Y: a -1/0/1 label matrix, where 0 means unknown
%   selectors: a cell array of function handles to use for prediciton
%   steps: the number of steps of prediction to run (-1 to go until end)
%   known: the initially known elements of the matrix
%   queryable: the elements that are allowed to be queried (default Y ~= 0)
%   C: slack variable in MMMF (default 1 XXX)
%
% OUTPUTS:
%   all_results: a cell array with one entry per selector, containing
%                cell arrays of num_known, rmse, pick [i,j], evaluations array

function [rmse] = get_rmse(X, selec)
    rmse = sqrt(sum((Y(Y ~= 0) - X(Y ~= 0)).^2) / numel(X))
end

if ~iscell(selectors); selectors = {selectors}; end
if nargin < 3; steps = -1; end
if nargin < 4; known = eye(size(Y)); end
if nargin < 5; queryable = (Y ~= 0); end
if nargin < 6; C = 1; end

addpath(genpath('yalmip'))
addpath('~/share/csdp/matlab')

known = logical(known);
[known_i, known_j] = find(known);
num_known_init = nnz(known);

queryable = logical(queryable);
queryable(known) = false;

Ytr_init = double(zeros(size(Y)));
Ytr_init(known) = Y(known);

% initial fit
[x_init, xu_init, xv_init] = solveD(Ytr_init, 'a', C);

all_results = cell(1, length(selectors));

for selector_i = 1 : length(selectors)
    selector = selectors{selector_i};
    Ytr = Ytr_init;
    x = x_init;
    xu = xu_init;
    xv = xv_init;
    num_known = num_known_init;
    can_query = queryable;

    results = cell(1, 4);
    results(1,:) = {num_known, get_rmse(sign(x)), [], []};

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
        [x, xu, xv] = solveD(Ytr, 'a', C);
        num_known = num_known + 1;

        % save results
        results(stepnum, :) = {num_known, get_rmse(x), [i,j], evals};
        stepnum = stepnum + 1;
    end

    all_results{selector_i} = results;
end
end
