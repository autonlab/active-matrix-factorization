function output = calloptiooqp(interfacedata)

% Author Johan Löfberg 
% $Id: callopticlp.m,v 1.6 2005-05-07 13:53:20 joloef Exp $

% Standard input interface
options = interfacedata.options;
F_struc = interfacedata.F_struc;
c       = interfacedata.c;
K       = interfacedata.K;
lb      = interfacedata.lb;
ub      = interfacedata.ub;
Q       = interfacedata.Q;

showprogress('Calling OOQP',options.showprogress);

blow = -inf(K.l,1);
if isempty(F_struc)  
    A = sparse([]);
    bupp = [];
    blow = [];
    Aeq = sparse([]);
    beq = [];
else
    Aeq = -F_struc(1:K.f,2:end);
    beq = F_struc(1:K.f,1);
    A =-F_struc(K.f+1:end,2:end);
    bupp =F_struc(K.f+1:end,1);
end

if isempty(lb)
    lb = -inf(length(c),1);
    ub = inf(length(c),1);
end

%Get Finite Element Indicies
ilb = double(isfinite(lb));
iub = double(isfinite(ub));
iblow = double(isfinite(blow));
ibupp = double(isfinite(bupp));
%Remove infinte values
lb(~ilb) = 0;
ub(~iub) = 0;
blow(~iblow) = 0;
bupp(~ibupp) = 0;

opts.tolfun = options.clp.primaltolerance;
opts.maxiter = options.clp.maxnumiterations;
opts.maxtime = options.clp.maxnumseconds;
opts.display = options.verbose;

H = 2*sparse(tril(Q));
if options.savedebug
    save ooqpdebug c H lb ilb ub iub Aeq beq A blow iblow bupp ibupp opts
end

solvertime = clock; 
[x,fval,stat,iter] = ooqp(full(c),H, full(lb),ilb,ub,iub,Aeq',full(beq),A',blow,iblow,full(bupp),ibupp,opts);
if interfacedata.getsolvertime solvertime = etime(clock,solvertime);else solvertime = 0;end

% No duals
D_struc = [];

switch stat
    case 1
        problem = 0;
    case 0
        problem = 3;
    case -1
        problem = 1;
    case -2
        problem = 11; 
    otherwise
        problem = -1;
end
infostr = yalmiperror(problem,'OOQP');       

% Save all data sent to solver?
if options.savesolverinput
    solverinput.A = A;
    solverinput.b = b;
    solverinput.f = c;
    solverinput.lb = lb;
    solverinput.ub = ub;    
else
    solverinput = [];
end

% Save all data from the solver?
if options.savesolveroutput
    solveroutput.x = x;
    solveroutput.fmin = fmin;
    solveroutput.flag = flag;
    solveroutput.output=output;
else
    solveroutput = [];
end

% Standard interface 
output.Primal      = x(:);
output.Dual        = D_struc;
output.Slack       = [];
output.problem     = problem;
output.infostr     = infostr;
output.solverinput = solverinput;
output.solveroutput= solveroutput;
output.solvertime  = solvertime;