%
% [xy,x,th,xu,xv,qval,Qval,Dval] = solveDord(y,maxoravg,C,
%     perrowthresh,requirethreshord,solver)
%
% Sets up and solves (using YALMIP) a MMMF problem on ordinal labels using
% the dual (the P version uses the primal, but don't try it on anything of
% substantial size---in fact, on anything at all).  For problems of
% any substantial size, it is highly recommended that printSDPAord
% and an external solver be used.
%
% Inputs:
%
% y = a matrix of labels.  Labels are (small) positive integers
% 1..R (the range is inferred from y).  0 indicates a missing label.
%
% maxoravg = 'a' for nuclear-norm, 'm' for max-norm (default is
% nuclear-norm)
% 
% C = label loss function
%   C>0: Shashua-Levin type hinge loss on immediate threshold, scaled by C
%   C<0: Hinge loss summed over all thresholds, scaled by (-C)
%   C=inf and C=-inf force hard margin constraints, but lead to a
%   different SDP
%   Default is C=inf, i.e. no slack, with constraints only on immediate
%   thresholds
%
% perrowthresh = 1 to allow different thresholds for each row. Default is
% 0, i.e. universal thresholds for entire matrix
% 
% requirethreshord = 1 to require thresholds be correctly ordered
% (especially when  C>0).
%
% solver = YALMIP solver to use (default is 'csdp').
%
% Outputs:
%
% xy = predicted labels for each entry
% x = learned real valued matrix
% th = threshold
%
% If perrowthresh=0 (default): th(  xy(i,a)-1) < x(i,a) < th(  xy(i,a))
% If perrowthresh=1          : th(i,xy(i,a)-1) < x(i,a) < th(i,xy(i,a))
% Where implicitly th(0)=th(i,0)=-inf and th(R)=th(i,R)=inf (these do not
% actually appear in the return th).  These inequalities also assume the
% thresholds are order: if C>0 but requirethreshord=0, craziness can occur.
%
% x = xu*xv, where xu and xv are low-norm (i.e. the learned U and
% V).
%
% qval = dual variables corresponding to label constraints
% Qval = dual variable as a matrix
% Dval = for max-norm, dual variables corresponding to bounding the norms.
% 
%
% Copyright May 2004, Nathan Srebro, nati@mit.edu
%


function [xy,x,th,xu,xv,qval,Qval,Dval] = solveDord(y,maxoravg,C,...
					  perrowthresh,requirethreshord,solver)
  if nargin<6
    solver = 'csdp';
  end
  if nargin<2
    maxoravg = 'a';
  end
  if nargin<3
    C = inf;
  end
  if C<0
    C = -C;
    sumrankmarg = 1;
  else
    sumrankmarg = 0;
  end
  if nargin<4
    perrowthresh = 0;
  end
  if nargin<5
    requirethreshord = 1;
  end
  if sumrankmarg == 1
    requirethreshord = 0;
    % achieved automatically
  end
  
  yalmip('clear');
  tic;

  [n,m]=size(y);
  [i,a,v] = find(y);
  p = length(i);
  R = max(v);  % maximal rank
  % vv(:,r) = 1 if y(i,a)>=r, i.e. X_ia should be on the right of
  % threshold between r and r+1, and -1 if it should be on left
  vv = sign(2*op(v,'>',1:(R-1))-1) ;
  
  % Dual variables corredponding to label constraints
  if sumrankmarg
    % q(r,ia) is dual variable corresponding to constraint of X_ia with
    % respect to threshold r and r+1.
    q = sdpvar((R-1),p,'full');
    qvv=q.*vv';
  else % max-rank-margin
    % q(1,ia) is dual variable corresponding to constraining X_ia to be
    % on the right of some threshold, and q(2,ia) corresponding to X_ia
    % having to be on the left.  If y(i,a)=1, X_ia doesn't have to be on
    % the left of right of anything, and if y(i,a)=R, X_ia doesn't have
    % to be on the right of anything.
    q = sdpvar(2,p,'full');
    q = q.*([v'>1;v'<R]);		% get rid of dual variables at extreems
    spreadq1 = sparse(v,1:p,q(1,:));	% (they don't get used, and
    spreadq2 = sparse(v,1:p,q(2,:));	% should be summed in objective)
    qvv = spreadq1(2:R,:)-spreadq2(1:(R-1),:);
    % EQUIV: Q = sparse(i,a,-0.5*( q(1,:).*(v>1)' - q(2,:).*(v<R)' ),n,m);
  end
  Q = sparse(i,a,-0.5*sum(qvv,1),n,m);
  c = set(q>=0,'labels');

  % Slack
  if (nargin>2) & (C<inf)
    c = c+set(q<=C,'slack');
  end

  % Dual variable corresponding to norm constraint
  if maxoravg == 'm' % max-norm
    D = sdpvar(1,n+m);
    QI = [sparse(1:n,1:n,D(1:n)) , Q ; ...
	  Q' , sparse(1:m,1:m,D(n+1:end))]
    c = c + set(sum(D)<=1,'traceQI');
  else  % avrage (sum) norm
    QI = [speye(n),Q; Q',speye(m)];
  end
  c = c + set(QI>=0,'QI');
  
  % Constraints corresponding to each threshold
  % for each r, sum vv(ia,r) q(r,ia) = 0
  if perrowthresh
    % sum duals corresponding to each row seperately
    threshas = sparse(i,1:p,1);
  else % Overall threshholds
       % sum all of them together
       threshas = ones(1,p);
  end
  numth = size(threshas,1); % =1, or =i if per-row
  threshsums = -threshas*qvv';
  if requirethreshord
    TO = sdpvar(numth,R-2,'full');
    c=c+set(TO>=0,'thord');
    c=c+set(threshsums==[TO zeros(numth,1)]-[zeros(numth,1) TO],'thresh');
  else 
    c=c+set(threshsums==0,'thresh');
  end
  
  c
  
  solvesdp(c,-sum(q(:)),sdpsettings('showprogress',1,'solver',solver));
  xx = dual(c('QI'));
  th = dual(c('thresh'));
  toc

  x = xx(1:n,(n+1):end);
  xy = sum(op(reshape(x,n,m,1),'>',reshape(th,[],1,R-1)),3)+1;
  if nargout>3
    [U,S,V] = svd(xx);
    U = U*sqrt(S);
    xu = U(1:n,:);
    xv = U((n+1):end,:);
  end
  
  if nargout>5
    qval=double(q);
  end
  
  if nargout>6
    Qval = double(Q);
    if (maxoravg=='m')
      Dval = double(D);
    end
  end
    

function out = op(arg1,operator,arg2)
  % computes the binnary operation on the arguments, extending 1-dim
  % dimmensions apropriately. E.g. it is ok to multiply 1xNxP and
  % MxNx1 matrices, subtarct a vector from a matrix, etc.
  %
  % Written by Nathan Srebro, MIT LCS, October 1998.
  
  shape1=[size(arg1),ones(1,length(size(arg2))-length(size(arg1)))] ;
  shape2=[size(arg2),ones(1,length(size(arg1))-length(size(arg2)))] ;
  
  out = feval(operator, ...
      repmat(arg1,(shape1==1) .* shape2 + (shape1 ~= 1) ), ...
      repmat(arg2,(shape2==1) .* shape1 + (shape2 ~= 1) )) ;

