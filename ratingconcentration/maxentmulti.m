function warmstart = maxentmulti(Xtr, mask, delta, warmstart,F, prior)

%


[N,M] = size(Xtr);

k = size(F,2);

settings = full(max(max(Xtr)));

if (~exist('warmstart','var') || isempty(warmstart))
    warmstart = zeros(2*(N+M)*k,1);
end

%[mu,nu] = computeaverages(Xtr,F);
[mu,nu,varu,varv] = computeaverages(Xtr,F);

C = sum(Xtr>0,2);
D = sum(Xtr>0)';
c = sum(mask,2);
d = sum(mask,1)';

[alpha, beta] = setbounds(c,d,C,D,delta); alpha = alpha*ones(1,k); beta = beta*ones(1,k);

mu(isnan(mu)) = 0;
nu(isnan(nu)) = 0;

options = optimset('GradObj','On','MaxFunEvals',1e16,...
    'MaxIter',100,'Display','final','DerivativeCheck','Off',...
    'TolFun',1e-7,...
    'TolX',1e-7,'LargeScale','on');
options.Algorithm = 'interior-point';
options.Hessian = {'lbfgs',3};
options.PlotFcns = {@optimplotx,@optimplotfval};
options.initBarrierParam = 1;
options.OutputFcn = @fminstatus;
%    'UseParallel','always',...
%    'ObjectiveLimit',-1e6,...

violations = checkconstraints3(warmstart,mu,nu,c,d,alpha,beta,mask,F,prior);

threshold = 1e-3;

[junk,worst] = sort(violations,'descend');

fprintf('Initial distribution produces %d constraints\n', nnz(violations>threshold));

active = warmstart>0;

active(violations>threshold) = 1;
fval = 0;

iters = 1;

lbfstatus;

change = 1;
global GLOBALITERS;
GLOBALITERS = 0;

maxiter = 500; % max iters per cut
total_maxiter = 3000;

while(iters<1 || (change>1e-3 && max(violations)>=threshold && ...
      ~isinf(fval) && ~isnan(fval) && iters*maxiter<total_maxiter))
    fprintf('Starting with %d constraints\n',nnz(active));
    
    auxdata{1}=mu;
    auxdata{2}=nu;
    auxdata{3} = c;
    auxdata{4} = d;
    auxdata{5} = alpha;
    auxdata{6} = beta;
    auxdata{7} = mask;
    auxdata{8} = F;
    auxdata{9} = active;
    auxdata{10} = prior;
    
    olditer = GLOBALITERS;

    % using lbfgsb for matlab
    X = lbfgsb(warmstart(active), zeros(nnz(active),1),...
        1e4*ones(nnz(active),1), 'objfunc', 'gradfunc', auxdata,...
        'lbfstatus','maxiter',maxiter,'factr',1e-9,'pgtol',1e-9);

    change = max(abs(warmstart(active)-X))
    
    warmstart(active) = X;
    
    violations = checkconstraints3(warmstart,mu,nu,c,d,alpha,beta,mask,F,prior);
       
    fprintf('Maximum active constraint violation = %f\n', ...
	    max(max(0,violations(active))));
    fprintf('Maximum inactive constraint violation = %f\n', ...
	    max(max(0,violations(~active))));
    
    fprintf('Optimization with %d constraints produced solution with %d violations, %d new\n',...
        nnz(active), nnz(violations>threshold), nnz(violations(~active)>threshold));
    fprintf('%d out of %d active constraints met\n', nnz(active.*(violations<threshold)),nnz(active));
    
    fprintf('Adding %d constraints. \n', nnz(violations(~active)>threshold));
    active(violations>threshold) = 1;
    
    iters = iters+1;

end

