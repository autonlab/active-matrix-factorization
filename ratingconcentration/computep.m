function [p,p2] = computep(mask,c,d,gammap,gamman,lambdap,lambdan,F,prior)

cbig = c*ones(1,size(gamman,2));
dbig = d*ones(1,size(lambdan,2));

U = ((gammap-gamman)./cbig)';
V = ((lambdap-lambdan)./dbig)';

U(isnan(U)) = 0;
V(isnan(V)) = 0;

settings = size(F,1);

p = ones(nnz(mask),settings);

for i=1:settings
    p(:,i) = nonzeros(spouterprod(mask,exp(F(i,:)*U), exp(F(i,:)*V)));
end

if (nnz(isinf(p)))
    p(isinf(p)) = realmax/settings;
end
if (nnz(isnan(p)))
    p(isnan(p)) = 0;
end
p(sum(p,2)==0,:) = eps;
p = p*diag(prior);

if (nargout==2)
    
    Umax = 0;
    Vmax = 0;
    
    for i=1:settings
        Umax = max(Umax, F(i,:)*U);
        Vmax = max(Vmax, F(i,:)*V);
    end
    %Umax = mean(F*U);
    %Vmax = mean(F*V);
    
    for i=1:settings
        p2(:,i) = nonzeros(spouterprod(mask,exp(F(i,:)*U-Umax), exp(F(i,:)*V-Vmax)));
    end
    p2 = p2*diag(prior);
    
    if (nnz(sum(p2,2)==0))
        p2(sum(p2,2)==0,:) = repmat(prior(:)',nnz(sum(p2,2)==0),1);
    end
    
end