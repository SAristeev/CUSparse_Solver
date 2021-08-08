#include "CUSparse_helper.h"

int main(){ 
    
    FILE* log;
    log = fopen("../output/CUSparse.log", "w");
    int rowsA, colsA, nnzA, * h_RowPtrA = NULL, * h_ColIndA = NULL, rowsB;
    double* h_ValA = NULL, * h_B = NULL, *h_X = NULL, *h_R, *fid_X = NULL;
    double start, stop, elapsedTime;
    
//==========================================================================
// Input
//==========================================================================

    start = second();

    csrMatrixFileInput("../input/A.txt", &rowsA, &nnzA, &h_RowPtrA, &h_ColIndA, &h_ValA);
    denseVectorFileInput("../input/B.vec", &rowsB, &h_B);
    denseVectorFileInput("../input/X.vec", &colsA, &fid_X);

    stop = second();
    elapsedTime = stop - start;
    fprintf(log, "input A, B, X            --- %10.6f sec\n", elapsedTime);
        
    if (rowsA != rowsB || rowsB != colsA) {
        printf("Wrong dimensions: rowsA = %d rowsB = %d rowX = %d", rowsA, rowsB, colsA);
        return -2;
    }

    fprintf(log, "inf-norm of |A|            =  %e\n", csr_mat_norminf(rowsA, colsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA));
    fprintf(log, "inf-norm of |B|            =  %e\n", vec_norminf(colsA, h_B));
    fprintf(log, "inf-norm of |Fidesys(X)|   =  %e\n", vec_norminf(colsA, fid_X));
    fprintf(log, "========================= \n");

//==========================================================================
// First Solver
//==========================================================================

    h_X = (double*)malloc(sizeof(double) * colsA);
    h_R = (double*)malloc(sizeof(double) * colsA);

    linearSolverSpSV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
                     rowsB, h_B,
                     colsA, h_X, log);


    fprintf(log, "inf-norm of |CUSparse(X)|  =  %e\n", vec_norminf(colsA, h_X));
    denseVectorFileOutput("../output/X_SpSv.vec", colsA, h_X);
    testFidesys(colsA, h_X, fid_X, log);
    testResidualSpMV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
                     rowsB, h_B,
                     colsA, h_X, log);
    fprintf(log, "inf-norm of |A*X - B|      =  %e\n", vec_norminf(colsA, h_X));
    fprintf(log, "========================= \n");

    free(h_X);
    free(h_R);
    h_X = NULL;
    h_R = NULL;

//==========================================================================
// Second Solver
//==========================================================================

    h_X = (double*)malloc(sizeof(double) * colsA);
    h_R = (double*)malloc(sizeof(double) * colsA);

    linearSolverBSRV2(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);

    fprintf(log, "inf-norm of |CUSparse(X)|  =  %e\n", vec_norminf(colsA, h_X));
    denseVectorFileOutput("../output/X_BSRV2.vec", colsA, h_X);
    testFidesys(colsA, h_X, fid_X, log);
    testResidualSpMV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);
    fprintf(log, "inf-norm of |A*X - B|      =  %e\n", vec_norminf(colsA, h_X));
    fprintf(log, "========================= \n");

    free(h_X);
    free(h_R);
    h_X = NULL;
    h_R = NULL;

//==========================================================================
// Third Solver
//==========================================================================

    h_X = (double*)malloc(sizeof(double) * colsA);
    h_R = (double*)malloc(sizeof(double) * colsA);

    linearSolverCHOL(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);

    fprintf(log, "inf-norm of |CUSparse(X)|  =  %e\n", vec_norminf(colsA, h_X));
    denseVectorFileOutput("../output/X_CHOL.vec", colsA, h_X);
    testFidesys(colsA, h_X, fid_X, log);
    testResidualSpMV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);
    fprintf(log, "inf-norm of |A*X - B|      =  %e\n", vec_norminf(colsA, h_X));
    fprintf(log, "========================= \n");
    free(h_X);
    free(h_R);
    h_X = NULL;
    h_R = NULL;

//==========================================================================
// Fourth Solver
//==========================================================================

    h_X = (double*)malloc(sizeof(double) * colsA);
    h_R = (double*)malloc(sizeof(double) * colsA);

    linearSolverLU(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);

    fprintf(log, "inf-norm of |CUSparse(X)|  =  %e\n", vec_norminf(colsA, h_X));
    denseVectorFileOutput("../output/X_LU.vec", colsA, h_X);
    testFidesys(colsA, h_X, fid_X, log);
    testResidualSpMV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
        rowsB, h_B,
        colsA, h_X, log);
    fprintf(log, "inf-norm of |A*X - B|      =  %e\n", vec_norminf(colsA, h_X));
    fprintf(log, "========================= \n");
    free(h_X);
    free(h_R);
    h_X = NULL;
    h_R = NULL;

    free(h_RowPtrA);
    free(h_ColIndA);
    free(h_ValA);
    free(h_B);
    free(fid_X);

    h_RowPtrA = NULL;
    h_ColIndA = NULL;
    h_ValA = NULL;
    h_B = NULL;
    fid_X = NULL;
    
    fclose(log);
    log = NULL;

	return 0;
}
