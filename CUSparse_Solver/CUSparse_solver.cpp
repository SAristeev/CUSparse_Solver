#include "CUSparse_helper.h"

void linearSolverSpSV(const int rowsA, const int nnzA,
    const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
    const int rowsB, const double* h_B,
    const int colsA, double* h_X, FILE* log) {

    int* d_RowPtrA, * d_ColIndA;
    double* d_ValA, * d_B, *d_X;
    double alpha = 1.0;
    double startAll, stopSolve, startSolve, stopAll;
    fprintf(log, "Standart SpSV-Solver\n");
    startAll = second();

//==========================================================================
// Device memory management
//==========================================================================

    CHECK_CUDA(cudaMalloc((void**)&d_RowPtrA, (rowsA + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ColIndA, nnzA * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ValA, nnzA * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_B, rowsB * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_X, colsA * sizeof(double)))
    
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
    fprintf(log, "Solver time              --- %10.6f sec\n", stopSolve - startSolve);

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
    fprintf(log, "All Solver CUDA timing   --- %10.6f sec\n", stopAll- startAll);
}


void linearSolverBSRV2(const int rowsA, const int nnzA,
    const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
    const int rowsB, const double* h_B,
    const int colsA, double* h_X, FILE* log) {

    int* d_RowPtrA, * d_ColIndA;
    double* d_ValA, * d_B, * d_X;
    double alpha = 1.0;
    double startAll, stopSolve, startSolve, stopAll;
    fprintf(log, "BSRV2(scrV2) - Solver BlockDim = 1\n");
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
    cusparseMatDescr_t descr = 0;
    bsrsv2Info_t info = 0;
    int pBufferSize;
    void* pBuffer = 0;
    int structural_zero, numerical_zero;

    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO))

    CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info))

//==========================================================================
// CUSAPRSE Analysis + Solve
//==========================================================================

    startSolve = second();

    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans, rowsA, nnzA, descr,
                                              d_ValA, d_RowPtrA, d_ColIndA, 1, info, &pBufferSize))
    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize))
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans, rowsA, nnzA, descr,
                                            d_ValA, d_RowPtrA, d_ColIndA, 1, 
                                            info, policy, pBuffer))

    int status = cusparseXbsrsv2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        fprintf(log, "L(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans, rowsA, nnzA, &alpha, descr,
                                         d_ValA, d_RowPtrA, d_ColIndA, 1, info,
                                         d_B, d_X, policy, pBuffer))

    status = cusparseXbsrsv2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        fprintf(log, "L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    stopSolve = second();
    fprintf(log, "Solver time              --- %10.6f sec\n", stopSolve - startSolve);
    
//==========================================================================
// CUSAPRSE APIs destroy
//==========================================================================

    
    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info))
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    
    CHECK_CUDA(cudaMemcpy(h_X, d_X, colsA * sizeof(double), cudaMemcpyDeviceToHost))

//==========================================================================
// Device memory deallocation
//==========================================================================

    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(d_RowPtrA))
    CHECK_CUDA(cudaFree(d_ColIndA))
    CHECK_CUDA(cudaFree(d_ValA))
    CHECK_CUDA(cudaFree(d_B))
    CHECK_CUDA(cudaFree(d_X))

    stopAll = second();
    fprintf(log, "All Solver CUDA timing   --- %10.6f sec\n", stopAll - startAll);
}

void linearSolverCHOL(const int rowsA, const int nnzA,
    const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
    const int rowsB, const double* h_B,
    const int colsA, double* h_X, FILE* log) {

    int* d_RowPtrA, * d_ColIndA;
    double* d_ValA, * d_B, * d_X, * d_Z, * h_Z = NULL;
    double alpha = 1.0;
    double startAll, stopSolve, startSolve, stopAll;

    fprintf(log, "Cholesky Solver \n");

    startAll = second();
    h_Z = (double*)malloc(rowsB * sizeof(double));
//==========================================================================
// Device memory management
//==========================================================================

    CHECK_CUDA(cudaMalloc((void**)&d_RowPtrA, (rowsA + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ColIndA, nnzA * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_ValA, nnzA * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_B, rowsB * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_X, colsA * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_Z, rowsB * sizeof(double)))
            
    CHECK_CUDA(cudaMemcpy(d_RowPtrA, h_RowPtrA, (rowsA + 1) * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_ColIndA, h_ColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_ValA, h_ValA, nnzA * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_B, h_B, rowsB * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_X, h_X, colsA * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_Z, h_Z, rowsB * sizeof(double), cudaMemcpyHostToDevice))

//==========================================================================
// CUSAPRSE APIs preparation
//==========================================================================

    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descr_M = 0, descr_L = 0;
    csric02Info_t info_M = 0;
    csrsv2Info_t info_L = 0, info_Lt = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_Lt, pBufferSize;
    void* pBuffer = 0;
    int structural_zero;
    int numerical_zero;

    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL; 
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

    CHECK_CUSPARSE(cusparseCreate(&handle))

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO))
    CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL))

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO))
    CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL))
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER))
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT))

    CHECK_CUSPARSE(cusparseCreateCsric02Info(&info_M))
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&info_L))
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&info_Lt))

//==========================================================================
// CUSAPRSE Analysis + Solve
//==========================================================================

    startSolve = second();

    CHECK_CUSPARSE(cusparseDcsric02_bufferSize(handle, rowsA, nnzA,
                                               descr_M, d_ValA, d_RowPtrA, d_ColIndA,
                                               info_M, &pBufferSize_M))
    CHECK_CUSPARSE(cusparseDcsrsv2_bufferSize(handle, trans_L, rowsA, nnzA,
                                              descr_L, d_ValA, d_RowPtrA, d_ColIndA,
                                              info_L, &pBufferSize_L))
    CHECK_CUSPARSE(cusparseDcsrsv2_bufferSize(handle, trans_Lt, rowsA, nnzA,
                                              descr_L, d_ValA, d_RowPtrA, d_ColIndA,
                                              info_Lt, &pBufferSize_Lt))

    pBufferSize = MAX(pBufferSize_M, MAX(pBufferSize_L, pBufferSize_Lt));

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize))

    CHECK_CUSPARSE(cusparseDcsric02_analysis(handle, rowsA, nnzA, descr_M,
                                             d_ValA, d_RowPtrA, d_ColIndA, info_M,
                                             policy_M, pBuffer))

    int status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        fprintf(log, "A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    CHECK_CUSPARSE(cusparseDcsrsv2_analysis(handle, trans_L, rowsA, nnzA, descr_L,
                                            d_ValA, d_RowPtrA, d_ColIndA,
                                            info_L, policy_L, pBuffer))
    CHECK_CUSPARSE(cusparseDcsrsv2_analysis(handle, trans_Lt, rowsA, nnzA, descr_L,
                                            d_ValA, d_RowPtrA, d_ColIndA,
                                            info_Lt, policy_Lt, pBuffer))
    CHECK_CUSPARSE(cusparseDcsric02(handle, rowsA, nnzA, descr_M,
                                    d_ValA, d_RowPtrA, d_ColIndA, 
                                    info_M, policy_M, pBuffer))

    status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        fprintf(log, "L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    CHECK_CUSPARSE(cusparseDcsrsv2_solve(handle, trans_L, rowsA, nnzA, &alpha, descr_L,
                                         d_ValA, d_RowPtrA, d_ColIndA, info_L,
                                         d_B, d_Z, policy_L, pBuffer))
    CHECK_CUSPARSE(cusparseDcsrsv2_solve(handle, trans_Lt, rowsA, nnzA, &alpha, descr_L,
                                         d_ValA, d_RowPtrA, d_ColIndA, info_Lt,
                                         d_Z, d_X, policy_Lt, pBuffer))

    stopSolve = second();
    fprintf(log, "Solver CUDA solve        --- %10.6f sec\n", stopSolve - startSolve);

//==========================================================================
// CUSAPRSE APIs destroy
//==========================================================================

    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyCsric02Info(info_M);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_Lt);
    cusparseDestroy(handle);

    CHECK_CUDA(cudaMemcpy(h_X, d_X, colsA * sizeof(double), cudaMemcpyDeviceToHost))

//==========================================================================
// Device memory deallocation
//==========================================================================

    CHECK_CUDA(cudaFree(d_RowPtrA))
    CHECK_CUDA(cudaFree(d_ColIndA))
    CHECK_CUDA(cudaFree(d_ValA))
    CHECK_CUDA(cudaFree(d_B))
    CHECK_CUDA(cudaFree(d_X))

    stopAll = second();
    fprintf(log, "All Solver CUDA timing   --- %10.6f sec\n", stopAll - startAll);
}