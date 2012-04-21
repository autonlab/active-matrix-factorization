function F = sets_square5(r)
% computes subset and quadratic features for rating values 1 through 5

F = zeros(1, 17);

F(r) = 1;
lists = nchoosek(1:5, 2);
membership = any(lists == r, 2);

F(6:15) = membership(:)';

F(16) = (r-1)/4;
F(17) = (r-1)^2/16;

