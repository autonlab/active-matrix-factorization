/* spouterprod.c
 * Bert Huang
 * bert@cs.columbia.edu
 *
 * This mex function provides a fast sparse outer product, which is 
 * useful when you need sparse entries in a large, dense outer product.
 * The input is a sparse mask, and full matrices U and V
 * The output is a sparse matrix with nonzeros where the mask is nonzero, 
 * and is equivelent to the matlab expression mask.*(U*V')
 *
 * Unfortunately matlab does not compute the above expression efficiently, 
 * so this mex file is helpful.
 *
 * Compile by typing 'mex spouterprod.c'
 *
 * Copyright 2009, 2010 Bert Huang
 * Feel free to contact the author with any questions or suggestions.
 *
 *
 * Updates:
 *
 * 5/27/10 - updated to support 64-bit matlab. Compile with -largeArrayDims
 *
 * Known Issues:
 *
 * - Code does not take advantage of multithreading
 * - Does not support sparse U and V
 *
 * This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize *ir, *jc, *ir_out, *jc_out;
    int nnz=0, M, N, D, i, j, 
            row, column;
    double *U, *V, *out, *data;
    mxArray *Ut, *Vt, *RHS[1], *LHS[1];
    
    if (nrhs != 3) 
        mexErrMsgTxt("Input error: Expected sparse mask, U, V, 1 output");
    if (!mxIsSparse(prhs[0]))
        mexErrMsgTxt("Error: Mask must be sparse\n");
    if (mxIsSparse(prhs[1]) || mxIsSparse(prhs[2]))
        mexErrMsgTxt("Error: sparse U or V is not implemented yet");
    
    /* load input */
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    M = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    nnz = jc[N];
    
    if (mxGetM(prhs[1]) == M && mxGetM(prhs[2]) == N) {
        /* transpose U and V for faster memory access */
        RHS[0] = prhs[1];
        mexCallMATLAB(1, LHS, 1, RHS, "transpose");
        Ut = LHS[0];
        RHS[0] = prhs[2];
        mexCallMATLAB(1, LHS, 1, RHS, "transpose");
        Vt = LHS[0];
        U = mxGetData(Ut);
        V = mxGetData(Vt);
        
        D = mxGetN(prhs[1]);
    } else if (mxGetN(prhs[1])==M && mxGetN(prhs[2])==N) {
        D = mxGetM(prhs[1]);
        U = mxGetPr(prhs[1]);
        V = mxGetPr(prhs[2]);
    } else 
        mexErrMsgTxt("Error: Matrix sizes are incorrect");
    
    /* open output */
    
    plhs[0] = mxCreateSparse(M, N, nnz, 0);
    out = mxGetPr(plhs[0]);
    
    ir_out = mxGetIr(plhs[0]);
    jc_out = mxGetJc(plhs[0]);
    
    column=0;
    jc_out[0] = jc[0]; 
    for (i=0; i<nnz; i++) {
        row = ir[i];
        ir_out[i] = ir[i];        
        
        while (i>=jc[column+1]) {
            column++;
        }
        
        /* printf("Nonzero at %d, %d\n", row,column); */
        
        out[i] = 0;
        for (j=0; j<D; j++) {
            out[i] += U[D*row+j]*V[D*column+j];
        }
        if (out[i]>1e128)
            out[i] = 1e128;
    }
    
    for (i=0; i<N; i++) {
        jc_out[i] = jc[i];
    }
    jc_out[N] = nnz;

}


