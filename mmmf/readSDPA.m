%
% [x,xu,xv,q,qq] = readSDPA(filename,n)
%
% Reads a solution in SDPA format (at least as produced by CSDP and
% DSDP) to a problem set up by printSDPA.  
%
% Inputs:
%
% filename = name of solution file (including all suffixes)
% n = number of rows in target matrix.  This MUST be provided as it
% cannot (easily) be infered from the SDP).
%
% Outputs:
%
% x = learned matrix, where sign(x) should agree (up to slack) with y
% x = xu*xv, where xu and xv are low-norm (i.e. the learned U and
% V).
%
% For CSDP:
% q = dual variables.  First those corresponding to constraints, and then,
% for max-norm problems, those corresponding to bounding the norm.
% 
% For DSDP:
% q = dual variables corresponding to constraints.
% qq = for max-norm, dual variables corresponding to max-norm constraint
%
% Copyright May 2004, Nathan Srebro, nati@mit.edu
%

function [x,xu,xv,q] = readSDPA(filename,n)
  fid = fopen(filename);
  qstring = fgetl(fid);
  q = -sscanf(qstring,'%f');
  if qstring(1)=='*'			% DSDP (or simmilar)
    fgetl(fid); fgetl(fid);             % skip first three lines
    blocksizes = sscanf(fgetl(fid),'%f'); % next line is block sizes
    qandqq = -sscanf(fgetl(fid),'%f');  % then the dual variables
    q = qandqq(1:blocksizes(2));
    qq = qandqq((blocksizes(2)+1):end);
  else                                  % CSDP (or simmilar)
    q = -sscanf(qstring,'%f');
  end
  v = reshape(fscanf(fid,'%f'),5,[]);
  fclose(fid);
  i = (v(1,:)==2) & (v(2,:)==1);
  YXXZ = sparse(v(3,i),v(4,i),v(5,i));
  x = full(YXXZ(1:n,(n+1):end));
  if nargout>1
    [U,S,V] = svd(full(YXXZ+YXXZ'-diag(diag(YXXZ))));
    US = U*sqrt(S);
    xu = US(1:n,:);
    xv = US((n+1):end,:);
  end
    