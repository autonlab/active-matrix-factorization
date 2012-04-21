function [alpha,beta] = setbounds(c,d,C,D,delta,delta2)

if (~exist('delta2','var'))
    delta2 = delta;
end

if (min(delta,delta2)>0)
    c(c==0) = eps;
    d(d==0) = eps;
    C(C==0) = eps;
    D(D==0) = eps;
    
    %alpha = sqrt(-1./(2.*C) * log(delta/2)) + sqrt(-(c+C)*log(delta/2)./(2.*C.*c));
    %beta = sqrt(-1./(2.*D) * log(delta2/2)) + sqrt(-(d+D)*log(delta2/2)./(2.*D.*d));
    
    alpha = (2-delta)*(sqrt(1./(2.*C)) + sqrt((c+C)./(2.*C.*c)));
    beta = (2-delta)*(sqrt(1./(2.*D)) + sqrt((d+D)./(2.*D.*d)));
    
    
    alpha(alpha>2) = 2;
    beta(alpha>2) = 2;
    
else
    alpha = 2*ones(size(c));
    beta = 2*ones(size(d));
end


