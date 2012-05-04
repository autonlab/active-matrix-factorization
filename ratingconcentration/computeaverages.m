function [mu,nu,varu,varv] = computeaverages(Xtr, F);

settings = 5; % max(nonzeros(Xtr)); % XXX - unhappy if Xtr doesn't include all values

k = size(F,2);

[N,M] = size(Xtr);

C = sum(Xtr>0,2);
D = sum(Xtr>0)';

mu = .5*ones(N,k);
nu = .5*ones(M,k);

[I,J,V] = find(Xtr);

vec = sparse(1:nnz(Xtr), V, ones(nnz(Xtr),1), nnz(Xtr),settings);

Fvec = vec*F;

if (nargout>2)
    varu = zeros(N,k);
    varv = zeros(M,k);
end

for i=1:k
    tmp = sparse(I,J,Fvec(:,i),N,M);
    
    mu(:,i) = sum(tmp,2)./C;
    nu(:,i) = sum(tmp,1)'./D;
   
    if (nargout>2)
        %compute variances too
        varu(:,i) = sum((tmp-sparse(I,J,mu(I,i),N,M)).^2,2)./C;
        varv(:,i) = sum((tmp-sparse(I,J,nu(J,i),N,M)).^2,1)'./D;
    end
end
