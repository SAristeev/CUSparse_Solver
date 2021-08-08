#include "CUSparse_helper.h"

void linearSolverSpSV(const int rowsA, const int nnzA,
    const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
    const int rowsB, const double* h_B,
    const int colsA, double* h_X) {

    int* d_RowPtrA, * d_ColIndA;
    double* d_ValA, * d_B, *d_X;
    double alpha = 1.0;
    double startAll, stopAll, elapsedAllTime, startSolve, stopSolve, elapsedSolveTime;

    startAll = second();

//==========================================================================
// Device memory management
//==========================================================================

    CHECK_CUDA( cudaMalloc((void**) &d_RowPtrA, (rowsA + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_ColIndA, nnzA * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_ValA, nnzA * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_B, rowsB * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_X, colsA * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(d_RowPtrA, h_RowPtrA, (rowsA + 1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ColIndA, h_ColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ValA, h_ValA, nnzA * sizeof(double), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_B, h_B, rowsB * sizeof(double), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_X, h_X, colsA * sizeof(double), cudaMemcpyHostToDevice) )

//==========================================================================
// CUSAPRSE APIs preparation
//==========================================================================

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecX;
    void* dBuffer = NULL;
    size_t               bufferSize = 0;
    cusparseSpSVDescr_t  spsvDescr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    //Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, rowsA, colsA, nnzA,
                                      d_RowPtrA, d_ColIndA, d_ValA,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // Create dense vector B
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecB, rowsB, d_B, CUDA_R_64F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, colsA, d_X, CUDA_R_64F) )
    // Create opaque data structure, that holds analysis data between calls.
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescr) )
//==========================================================================
// CUSAPRSE Analysis + Solve
//==========================================================================

    startSolve = second();

    CHECK_CUSPARSE( cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecB, vecX, CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                            &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, vecB, vecX, CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                          dBuffer) )
    CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecB, vecX, CUDA_R_64F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) )

    stopSolve = second();

//==========================================================================
// CUSAPRSE APIs destroy
//==========================================================================

    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA( cudaMemcpy(h_X, d_X, colsA * sizeof(double), cudaMemcpyDeviceToHost) )

//==========================================================================
// Device memory deallocation
//==========================================================================
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_RowPtrA) )
    CHECK_CUDA( cudaFree(d_ColIndA) )
    CHECK_CUDA( cudaFree(d_ValA) )
    CHECK_CUDA( cudaFree(d_B) )
    CHECK_CUDA( cudaFree(d_X) )
    stopAll = second();
    elapsedAllTime = stopAll - startAll;
    elapsedSolveTime = stopSolve - startSolve;
    printf("All   CUDA timing: = %10.6f sec\n", elapsedAllTime);
    printf("Solve CUDA timing: = %10.6f sec\n", elapsedSolveTime);
}