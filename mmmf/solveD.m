%
% [x,xu,xv,Qval,Dval] = solveD(y,maxoravg,C,solver)
%
% Sets up and solves (using YALMIP) a MMMF problem on binary labels
% using the dual (the P version uses the primal, but don't try it on
% anything of substantial size).  This is limited by YALMIP, and on
% larger problems it might be better to use printSDPA and an
% external solver.
%
% Inputs:
%
% y = a matrix of +1/0/-1 labels, 0 being no (missing) label.
%
% maxoravg = 'a' for nuclear norm, 'm' for max-norm (default is
% nuclear norm)
%
% C = coefficient for slack penalty. default = inf (no slack)
%
% solver = the YALMIP solver to use (default is 'csdp')
%
% Outputs:
%
% x = learned matrix, where sign(x) should agree (up to slack) with y
% x = xu*xv, where xu and xv are low-norm (i.e. the learned U and V).
%
% Qval = dual variables corresponding to constraints (analogous to
% alphas in SVM).  You don't explicitly need them, but they are useful
% for, e.g., deciding on C (as C is a bound on Qval) or
% understanding the solution.
%
% Dval = for max-norm, dual variables corresponding to bounding the norms.
% 
%
% Copyright May 2004, Nathan Srebro, nati@mit.edu
%

function [x,xu,xv,Qval,Dval] = solveD(y,maxoravg,C,solver)
  if nargin<4
    solver = 'csdp';
  end
  if nargin<3
    C = inf;
  end
  if nargin<2
    maxoravg = 'a';
  end
  yalmip('clear');
  tic;
  [n,m]=size(y);
  [i,a,v] = find(y);
  q = sdpvar(1,length(v));
  Q = sparse(i,a,-0.5*q.*v',n,m);
  c = set(q>=0);
  if maxoravg ~= 'm'
    QI = [speye(n),Q; Q',speye(m)];
    % QI = sparse([1:(n+m),i',n+a'], [1:(n+m),n+a',i'], [ones(1,n+m),repmat(-0.5*q.*v',1,2)], n+m, n+m);
  else % max-norm
    D = sdpvar(1,n+m);
    QI = sparse([1:(n+m),i',n+a'], [1:(n+m),n+a',i'], ...
		[D, repmat(-0.5*q.*v',1,2)], n+m, n+m);
    c = c + set(sum(D)<=1);
  end
  c = c + set(QI>=0,'QI');
  if (nargin>2) & (C<inf)
    c = c+set(q<=C);
  end
  settings = sdpsettings('showprogress', 1, 'solver', solver, 'cachesolvers', 1);
  d = solvesdp(c, -sum(q), settings);
  num_runs = 1;
  while d.problem ~= 0 & ~strcmp(d.info(1:18), 'Numerical problems')
      disp(d.info);
      if num_runs > 5
          error()
      end
      % hackety hack
      C = C * (1 + randn() * .1);
      d = solvesdp(c, -sum(q), settings);
      num_runs = num_runs + 1;
  end
  xx = dual(c('QI'));
  x = xx(1:n,(n+1):end);
  toc
  if nargout>1
    [U,S,V] = svd(xx);
    U = U*sqrt(S);
    xu = U(1:n,:);
    xv = U((n+1):end,:);
  end
  if nargout>3
    Qval = double(Q);
  end
  if (nargout>4) & (maxoravg=='m')
    Dval = double(D);
  end


