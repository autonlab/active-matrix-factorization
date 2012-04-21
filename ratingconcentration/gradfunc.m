function df = gradfunc(X,auxdata)


mu = auxdata{1};
nu = auxdata{2};
c = auxdata{3};
d = auxdata{4};
alpha = auxdata{5};
beta = auxdata{6};
mask = auxdata{7};
F = auxdata{8};
active = auxdata{9};

if (length(auxdata)==10)
    prior = auxdata{10};
else
    prior = .2*ones(5,1);
end

[f,df] = dual3(X,mu,nu,c,d,alpha,beta,mask,F,active,prior);
