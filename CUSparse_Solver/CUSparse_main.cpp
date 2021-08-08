#include "CUSparse_helper.h"

int main(){ 
    int rowsA, colsA, nnzA, * h_RowPtrA = NULL, * h_ColIndA = NULL, rowsB;
    double* h_ValA = NULL, * h_B = NULL, *h_X = NULL, *h_R, *fid_X = NULL;
    csrMatrixFileInput("../input/A.txt", &rowsA, &nnzA, &h_RowPtrA, &h_ColIndA, &h_ValA);
    denseVectorFileInput("../input/B.vec", &rowsB, &h_B);
    denseVectorFileInput("../input/X.vec", &colsA, &fid_X);

    h_X = (double*)malloc(sizeof(double) * colsA);
    h_R = (double*)malloc(sizeof(double) * colsA);

    if (rowsA != rowsB || rowsB != colsA) {
        printf("Wrong dimensions: rowsA = %d rowsB = %d rowX = %d", rowsA, rowsB, colsA);
        return -2;
    }

    linearSolverSpSV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
                     rowsB, h_B,
                     colsA, h_X);
    printf("inf-norm of |A| =  %e\n", csr_mat_norminf(rowsA, colsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA));
    printf("inf-norm of |B| =  %e\n", vec_norminf(colsA, h_B));
    printf("inf-norm of |Fidesys(X)| =  %e\n", vec_norminf(colsA, fid_X));
    printf("inf-norm of |CUSparse(X)| =  %e\n", vec_norminf(colsA, h_X));
    
    denseVectorFileOutput("../output/X.vec", colsA, h_X);

    testFidesys(colsA, h_X, fid_X);
    testResidualSpMV(rowsA, nnzA, h_RowPtrA, h_ColIndA, h_ValA,
                     rowsB, h_B,
                     colsA, h_X);

    printf("inf-norm of |A*X - B| =  %e\n", vec_norminf(colsA, h_X));

    free(h_RowPtrA);
    free(h_ColIndA);
    free(h_ValA);
    free(h_X);
    free(h_B);
    free(h_R);
    free(fid_X);

    h_RowPtrA = NULL;
    h_ColIndA = NULL;
    h_ValA = NULL;
    h_X = NULL;
    h_B = NULL;
    h_R = NULL;
    fid_X = NULL;

	return 0;
}
