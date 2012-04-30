#include "mex.h"



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize *ir, *jc;
    int nnz=0, M, N, D, i, j, 
            row, column;
    double  *rowsum, *colsum, *E;
    
    if (nrhs != 2) 
        mexErrMsgTxt("Input error: Expected sparse mask, E, 1 output");
    if (!mxIsSparse(prhs[0]))
        mexErrMsgTxt("Error: Mask must be sparse\n");
    if (mxIsSparse(prhs[1]))
        mexErrMsgTxt("Error: sparse E is not implemented yet");
    
    /* load input */
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    M = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    nnz = jc[N];
    D = mxGetN(prhs[1]);
    
    if (mxGetM(prhs[1]) != nnz)
        mexErrMsgTxt("Num rows of E must be equal to nnz(mask)");
    
    E = mxGetPr(prhs[1]);
    /* open rowsum */
    plhs[0] = mxCreateDoubleMatrix(M, D, mxREAL);
    rowsum = mxGetPr(plhs[0]);
    /* open colsum  */
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
            rowsum[j*M+row] += E[i+nnz*j];
            colsum[j*N+column] += E[i+nnz*j];
        }
    }
}


