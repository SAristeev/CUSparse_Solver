#include "CUSparse_helper.h"

void testResidualSpMV(const int rowsA, const int nnzA,
    const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
    const int rowsB, const double* h_B,
    const int colsA, double* h_X, FILE* log) {
    
    int* d_RowPtrA, * d_ColIndA;
    double* d_ValA, * d_B, * d_X;
    double alpha = 1.0, beta = -1.0;
    double startAll, stopAll;

    startAll = second();

//==========================================================================
// Device memory management
//==========================================================================

    CHECK_CUDA(cudaMalloc((void**)&d_RowPtrA, (rowsA + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ColIndA, nnzA * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ValA, nnzA * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_B, rowsB * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_X, colsA * sizeof(double)))

    CHECK_CUDA(cudaMemcpy(d_RowPtrA, h_RowPtrA, (rowsA + 1) * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_ColIndA, h_ColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_ValA, h_ValA, nnzA * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_B, h_B, rowsB * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_X, h_X, colsA * sizeof(double), cudaMemcpyHostToDevice))

//==========================================================================
// CUSAPRSE APIs preparation
//==========================================================================

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecX;
    void* dBuffer = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    //Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rowsA, colsA, nnzA,
                                     d_RowPtrA, d_ColIndA, d_ValA,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
    // Create dense vector B
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, rowsB, d_B, CUDA_R_64F))
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, colsA, d_X, CUDA_R_64F))
    
//==========================================================================
// CUSAPRSE Analysis + Multiply
//==========================================================================

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecB, CUDA_R_64F,
                                           CUSPARSE_MV_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecB, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer))

//==========================================================================
// CUSAPRSE APIs destroy
//==========================================================================

    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaMemcpy(h_X, d_X, colsA * sizeof(double), cudaMemcpyDeviceToHost))

//==========================================================================
// Device memory deallocation
//==========================================================================

    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(d_RowPtrA))
    CHECK_CUDA(cudaFree(d_ColIndA))
    CHECK_CUDA(cudaFree(d_ValA))
    CHECK_CUDA(cudaFree(d_B))
    CHECK_CUDA(cudaFree(d_X))
    stopAll = second();
    
    //fprintf(log, "Residual  A*X - B timing --- %10.6f sec\n", stopAll - startAll);
}