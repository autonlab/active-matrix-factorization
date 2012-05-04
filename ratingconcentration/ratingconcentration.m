function [E, P, vals, lagrange] = ratingconcentration(X, mask, featureFunc, delta, warmstart, vals)
% function [E, P] = ratingconcentration(I,J,F)
%
% E = sparse matrix of expected scores, assuming function values F are
%     ordinal (e.g., 1 through 5)
% P = multinomial distribution for each entry in mask
% vals = list of rating values
% lagrange = lagrange multipliers (can be used to warm start different runs)
%
% X = input sparse rating matrix
% mask = sparse 0-1 (or logical) matrix marking query entries
% featureFunc = function handle that maps a rating value to a vector of
%               feature values
% delta = regularization parameter between 0 and 2 (2 means match
%         expectations perfectly, 0 means don't match expectations at all)
% warmstart = lagrange multipliers from previous run for warm starting
% vals = unique available values (useful if X doesn't contain them all);
%        should be a column vector

% this path should be changed to wherever lbfgs-for-matlab is stored
addpath('lbfgsb-for-matlab');


%% initialize
[M, N] = size(X);

if nargin < 6
    vals = unique(nonzeros(X));
else
    vals = unique([nonzeros(X); vals]);
end
settings = length(vals);

F = zeros(length(vals), length(featureFunc(vals(1))));
for i = 1:length(vals)
    F(i,:) = featureFunc(vals(i));
end

% count of query entries
c = sum(mask>0,2);
d = sum(mask>0)';

%% compute prior

fprintf('Prior for ratings: ');
prior = zeros(settings,1);
for i=1:settings
    prior(i) = nnz(X==i)/nnz(X);
    fprintf('%d: %f\t', i, prior(i));
end
fprintf('\n');

fprintf('Trying delta = %f\n', delta);

%% optimize LaGrange multipliers
if ~exist('warmstart', 'var')
    warmstart = [];
end
lagrange = maxentmulti(X, mask, delta, warmstart, F, prior);

k = size(F,2);
gammap = reshape(lagrange(1:M*k),M,k);
gamman = reshape(lagrange(M*k+1:2*M*k),M,k);
lambdap = reshape(lagrange(2*M*k+1:2*M*k+N*k),N,k);
lambdan = reshape(lagrange(2*M*k+N*k+1:end), N,k);

[~,P] = computep(mask+(X>0),c,d,gammap,gamman,lambdap,lambdan,F,prior);
Z = sum(P,2);
% set numerically-zero probabilities to prior
P(Z<realmin,:) = repmat(prior',nnz(Z<realmin),1);
Z(Z<realmin,:) = sum(P(Z<realmin,:),2);

[I,J] = find(mask+(X>0));

pexpvec = (P*vals(:))./Z;
E = sparse(I,J,pexpvec,M,N);


