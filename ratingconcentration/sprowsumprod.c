#include "mex.h"



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize *ir, *jc;
    int nnz=0, M, N, D, i, j, x, 
            row, column, settings;
    double  *rowsum, *colsum, *p, *F;
    
    if (nrhs != 3) 
        mexErrMsgTxt("Input error: Expected sparse mask, p, F, 1 output");
    if (!mxIsSparse(prhs[0]))
        mexErrMsgTxt("Error: Mask must be sparse\n");
    if (mxIsSparse(prhs[1]) || mxIsSparse(prhs[2]))
        mexErrMsgTxt("Error: sparse p or F is not implemented yet");
    
    /* load input */
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    M = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    nnz = jc[N];
    
    if (mxGetM(prhs[1]) != nnz)
        mexErrMsgTxt("Num rows of p must be equal to nnz(mask)");
    settings = mxGetN(prhs[1]);
    
    if (mxGetM(prhs[2]) != settings)
        mexErrMsgTxt("Num rows of F must be equal to num cols of p");
    D = mxGetN(prhs[2]);
    
    p = mxGetPr(prhs[1]);
    F = mxGetPr(prhs[2]);
    /* open rowsum */
    plhs[0] = mxCreateDoubleMatrix(M, D, mxREAL);
    rowsum = mxGetPr(plhs[0]);
    /* open colsum */
    plhs[1] = mxCreateDoubleMatrix(N, D, mxREAL);
    colsum = mxGetPr(plhs[1]);
    
    column=0;
    for (i=0; i<nnz; i++) {
        row = ir[i];
        
        while (i>=jc[column+1]) {
            column++;
        }
        
        /* printf("Nonzero at %d, %d\n", row,column); */
        
        for (j=0; j<D; j++) {
            for (x=0; x<settings; x++) {
                rowsum[j*M+row] += p[i+nnz*x]*F[j*settings+x];
                colsum[j*N+column] += p[i+nnz*x]*F[j*settings+x];
            }
        }
    }

}


