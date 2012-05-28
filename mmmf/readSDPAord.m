%
% [xy,x,th,xu,xv,q,qq] = readSDPAord(filename,n,R)
%
% Reads a solution in SDPA format (at least as produced by CSDP and
% DSDP) to a problem set up by printSDPAord.  
%
% Inputs:
%
% filename = name of solution file (including all suffixes)
% n = number of rows in target matrix.
% R = number of ranking levels, i.e. maximal rank in training data
% Both n and R MUST be provided as they cannot (easily) be inferred from the SDP.
%
% Outputs:
%
% xy = predicted labels for each entry (integers in the range 1..R)
% x = learned real-valued matrix
% th = learned threshold
% x = xu*xv, where xu and xv are low-norm (i.e. the learned U and V).
%
% If perrowthresh=0 (default): th(  xy(i,a)-1) < x(i,a) < th(  xy(i,a))
% If perrowthresh=1          : th(i,xy(i,a)-1) < x(i,a) < th(i,xy(i,a))
% Where implicitly th(0)=th(i,0)=-inf and th(R)=th(i,R)=inf (these do not
% actually appear in the return th).  These inequalities also assume the
% thresholds are order: if C>0 but requirethreshord=0, craziness can occur.
%
% CSDP:
% q = dual variables.  First those corresponding to constraints, and then,
% for max-norm problems, those corresponding to bounding the norm, and
% finally, those corresponding to threshold ordering requirements.
%
% DSDP:
% q = dual variables corresponding to constraints
% qq = other dual variables
%
% Copyright May 2004, Nathan Srebro, nati@mit.edu
%

function [xy,x,th,xu,xv,q,qq] = readSDPAord(filename,n,R)
  fid = fopen(filename);
  qstring = fgetl(fid);
  if qstring(1)=='*'			% DSDP
    fgetl(fid); fgetl(fid); % skip first three lines
    blocksizes = sscanf(fgetl(fid),'%f'); % next line is block sizes
    qandqq = -sscanf(fgetl(fid),'%f');
    q = qandqq(1:blocksizes(4));
    qq = qandqq((blocksizes(4)+1):end);
  else
    q = -sscanf(qstring,'%f');
  end
  v = reshape(fscanf(fid,'%f'),5,[]);
  fclose(fid);
  i = (v(1,:)==2) & (v(2,:)==1);
  YXXZ = sparse(v(3,i),v(4,i),v(5,i));
  x = full(YXXZ(1:n,(n+1):end));
  bias = v(5,(v(1,:)==2)&(v(2,:)==2));
  thi = ((v(1,:)==2) & (v(2,:)==3));
  th = reshape(v(5,thi),R-1,[])'-bias;
  xy = threshold(x,th);
  if nargout>1
    [U,S,V] = svd(full(YXXZ+YXXZ'-diag(diag(YXXZ))));
    US = U*sqrt(S);
    xu = US(1:n,:);
    xv = US((n+1):end,:);
  end
    
  
function y = threshold(x,th)
  y = sum(op(x,'>',reshape(th,size(th,1),1,[])),3)+1;

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

  
  