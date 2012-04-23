function [h] = select_biggest_constraint_violation()
% supposed to be selection based on expected collapse in hull size
% will figure out details later...
    function [i, j, evals] = inner(Xtr, mask, P, ~, vals, featureFunc, ~, delta)
        [M, N] = size(mask);
        [I, J] = find(mask);
        
        % figure out the alpha, beta bounds
        F = zeros(length(vals), length(featureFunc(vals(1))));
        for i = 1:length(vals)
            F(i,:) = featureFunc(vals(i));
        end

        C = sum(Xtr>0, 2);
        D = sum(Xtr>0)';
        c = sum(mask, 2);
        d = sum(mask, 1)';

        [alpha, beta] = setbounds(c, d, C, D, delta);
        alpha = alpha*ones(1,k);
        beta = beta*ones(1,k);
        
        % figure out total expected bounds violation for each value
        

        evals = sparse(I, J, sum(P(mask(:), vals >= cutoff), 2), M, N);
        
        % pick the highest one
        [~, idx] = max(evals(:));
        [i, j] = ind2sub([M,N], idx);
    end

    h = @inner;
end