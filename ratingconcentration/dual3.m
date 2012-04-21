function [f,df] = dual3(X,mu,nu,c,d,alpha,beta,mask,F,active,prior)

[N,M] = size(mask);
k = size(mu,2);
settings = size(F,1);

if (exist('active','var'))
    %X = full(sparse(find(active),ones(nnz(active),1),X,2*(N+M)*k,1));
    %%slower version
    X = accumarray([find(active) ones(nnz(active),1)],X,[2*(N+M)*k,1]);
end

gammap = reshape(X(1:N*k),N,k);
gamman = reshape(X(N*k+1:2*N*k),N,k);
lambdap = reshape(X(2*N*k+1:2*N*k+M*k),M,k);
lambdan = reshape(X(2*N*k+M*k+1:end), M,k);

% all variables are set up.

f = 0;

f = f-sum(sum((gammap-gamman).*mu));
f = f-sum(sum((lambdap-lambdan).*nu));

%alphabig = alpha*ones(1,k);
%betabig = beta*ones(1,k);
alphabig = alpha;
betabig = beta;


%%%%% no deviation allowed at mu = 1 or 0


%flattenbounds

%%%%%%%%%%%%%%

f = f+sum(sum((gammap+gamman).*alphabig));
f = f+sum(sum((lambdap+lambdan).*betabig));

c(c==0) = eps;
d(d==0) = eps;

p = computep(mask,c,d,gammap,gamman,lambdap,lambdan,F,prior);

Z = sum(p,2);

f = f+sum(log(Z));

if (nargout>1)
    %pnorm = p'*spdiags(1./Z,0,nnz(mask),nnz(mask));
    for i=1:settings
        p(:,i) = p(:,i)./Z;
    end
    clear Z;
  
    %E = p*F;
    
    %[rowsum, colsum] = sprowcolsum(mask,E);
    
    [rowsum, colsum] = sprowsumprod(mask,p,F);
   
    dgammap = -mu + alphabig;
    dgamman = mu + alphabig;
    dlambdap = -nu + betabig;
    dlambdan = nu + betabig;
    
    cbig = c*ones(1,size(gamman,2));
    dbig = d*ones(1,size(lambdan,2));
    
    dgammap = dgammap + 1./cbig .* rowsum;
    dgamman = dgamman - 1./cbig .* rowsum;
    dlambdap = dlambdap + 1./dbig .* colsum;
    dlambdan = dlambdan - 1./dbig .* colsum;
    
    df = [dgammap(:); dgamman(:); dlambdap(:); dlambdan(:)];
   
    df(isnan(df)) = 0;

    if (exist('active','var'))
        df = df(active);
    end
end
