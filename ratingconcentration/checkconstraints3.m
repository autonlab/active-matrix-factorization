function violations = checkconstraints3(X,mu,nu,c,d,alpha,beta,mask,F,prior)

%% set up figures
%persistent lambdafig;
%persistent devfig;
%if (isempty(lambdafig) || ~ishandle(lambdafig))
%    figure(38); subplot(211); lambdafig = gca;
%end
%if (isempty(devfig) || ~ishandle(devfig))
%    figure(38); subplot(212); devfig = gca;
%end

c(c==0) = eps; d(d==0) = eps;

if (~exist('prior','var'))
    prior = 1;
end

[N,M] = size(mask);
k = size(mu,2);

settings = size(F,1);

gammap = reshape(X(1:N*k),N,k);
gamman = reshape(X(N*k+1:2*N*k),N,k);
lambdap = reshape(X(2*N*k+1:2*N*k+M*k),M,k);
lambdan = reshape(X(2*N*k+M*k+1:end), M,k);

% all variables are set up.

%alphabig = alpha*ones(1,k);
%betabig = beta*ones(1,k);
alphabig = alpha;
betabig = beta;

c = sum(mask,2);
d = sum(mask,1)';

p = computep(mask,c,d,gammap,gamman,lambdap,lambdan,F,prior);

Z = sum(p,2);
p(Z<realmin,:) = repmat(prior',nnz(Z<realmin),1);
Z(Z<realmin,:) = sum(p(Z<realmin,:),2);

E = full(sparse(F)'*p'*spdiags(1./Z,0,nnz(mask),nnz(mask)))';

[rowsum, colsum] = sprowcolsum(mask,E);

cbig = c*ones(1,size(gamman,2));
dbig = d*ones(1,size(lambdan,2));

rowavg =  (1./cbig) .* (rowsum.*(cbig>eps));
colavg = (1./dbig) .* (colsum.*(dbig>eps));

violations = [mu(:)-rowavg(:) - alphabig(:);...
    rowavg(:)-mu(:) - alphabig(:);...
    nu(:)-colavg(:) - betabig(:);
    colavg(:)-nu(:) - betabig(:)];...
    
%if (nargout==0)

%figure('Name','Constraint Violations','NumberTitle','off')

%plot(lambdafig,X);
%title(lambdafig,'lambda values');

%plot(devfig,max(violations,0));
%ylabel(devfig,'deviation');
%xlabel(devfig,'row/column feature');
%ax = axis(devfig);
%axis(devfig,[0 length(violations) 0 ax(4)]);
%drawnow;
%end




