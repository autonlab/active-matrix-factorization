Collaborative Filtering via Rating Concentration

This is the software used in the paper 
"Collaborative Filtering via Rating Concentration"
by Bert Huang and Tony Jebara. 

@inproceedings{HuaJeb10,
	Author = {Huang, B. and Jebara, T.},
	Booktitle = {Proceedings of the 13th International Conference on Artificial Intelligence and Statistics},
	Month = {May},
	Title = {Collaborative Filtering via Rating Concentration},
	Year = {2010}}

See demo.m to see how the software is called. The mex files are compiled
for 64-bit Mac OS X, and if you are on a different platform, you should call
makeMex.m to compile my mex files and follow the instructions inside
the lbfgs-for-matlab directory to compile lbfgs on your platform 
(or http://www.mathworks.com/matlabcentral/fileexchange/authors/28250).
This will require a fortran compiler and probably a lot of frustration.
Alternatively, you can edit the line of code in (86) maxentmulti.m that
calls lbfgs and replace it with your favorite nonlinear optimizer. 

The demo runs on a 50/50 split of the movielens million data set 
(movielens.org). 


