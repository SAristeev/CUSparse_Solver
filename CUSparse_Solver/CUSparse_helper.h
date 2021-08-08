#include <stdlib.h>           // EXIT_FAILURE
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#ifndef __CUSPARSE_HELPER_H__
#define __CUSPARSE_HELPER_H__

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#define CHECK_CUDA(func)                                                      \
{                                                                             \
    cudaError_t status = (func);                                              \
    if (status != cudaSuccess) {                                              \
        printf("CUDA API failed at line %d with error: %s (%d)\n",            \
               __LINE__, cudaGetErrorString(status), status);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

#define CHECK_CUSPARSE(func)                                                  \
{                                                                             \
    cusparseStatus_t status = (func);                                         \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cusparseGetErrorString(status), status);             \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

void csrMatrixFileInput(char const* FileName, int* rowsA, int* nnzA, int** h_RowPtrA, int** h_ColIndA, double** h_ValA);
void denseVectorFileInput(char const* FileName, int* rowsA, double** h_ValA);
void denseVectorFileOutput(char const* FileName, int rowsA, const double* h_ValA);

void linearSolverSpSV(const int rowsA, const int nnzA,
                      const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
                      const int rowsB, const double* h_B,
                      const int rowsX, double* h_X, FILE *log);
void linearSolverBSRV2(const int rowsA, const int nnzA,
                      const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
                      const int rowsB, const double* h_B,
                      const int rowsX, double* h_X, FILE* log);
void linearSolverCHOL(const int rowsA, const int nnzA,
                      const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
                      const int rowsB, const double* h_B,
                      const int rowsX, double* h_X, FILE* log);

void testResidualSpMV(const int rowsA, const int nnzA,
                      const int* h_RowPtrA, const int* h_ColIndA, const double* h_ValA,
                      const int rowsB, const double* h_B,
                      const int rowsX, double* h_X, FILE* log);
double vec_norminf(int n, const double* x);
double csr_mat_norminf(int rowsA, int colsA, int nnzA, const int* RowPtrA, const int* ColIndA, const double* ValA);
double second(void);
void testFidesys(const int colsA, const double* h_X, const double* fid_X, FILE* log);
#endif
