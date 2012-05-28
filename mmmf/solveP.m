%
% [x,xu,xv] = solveP(y,maxoravg,C,solver)
%
% Sets up and solves (using YALMIP) a MMMF problem on binary
% labels.  This version sets up the primal SDP.  It is provided for
% reference purpose: it is allways better to use solveD or
% printSDPA and an external solver.
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
% Copyright May 2004, Nathan Srebro, nati@mit.edu
%


function [x,xu,xv] = solveP(y,maxorsum,C,solver)
  if nargin<4
    solver = 'csdp';
  end
  if nargin<3
    C = inf;
  end
  if nargin<2
    maxorsum = 's';
  end
  yalmip('clear');
  tic;
  [n,m]=size(y);
  X = sdpvar(n,m,'full');
  Y = sdpvar(n);
  Z = sdpvar(m);
  S = (y~=0);
  YXXZ = [Y X;X' Z];
  c = set(YXXZ>=0);
  if maxorsum == 's' % sum-norm
    obj = trace(Y)+trace(Z);
  else % max-norm
    obj = sdpvar(1);
    c = c + set(diag(Y)<=obj)+set(diag(Z)<=obj);
  end
  if C<inf
    e = sdpvar(nnz(S),1);
    c = c+set(e>=0);
    obj = obj+C*sum(e);
  else
    e = 0;
  end
  c = c+set(y(S).*X(S)>=1-e);
  solvesdp(c,obj,sdpsettings('showprogress',1,'solver',solver));
  x = double(X);
  toc
  if nargout>1
    [U,S,V] = svd(double(YXXZ));
    U = U*sqrt(S);
    xu = U(1:n,:);
    xv = U((n+1):end,:);
  end
   
