#include "CUSparse_helper.h"
#include <math.h>


double vec_norminf(int n, const double* x)
{
    double norminf = 0;
    for (int j = 0; j < n; j++) {
        double x_abs = fabs(x[j]);
        norminf = (norminf > x_abs) ? norminf : x_abs;
    }
    return norminf;
}


double csr_mat_norminf(int rowsA, int colsA, int nnzA,
                       const int* RowPtrA, const int* ColIndA, const double* ValA){
    double norminf = 0;
    for (int i = 0; i < colsA; i++) {
        double sum = 0.0;
        const int start = RowPtrA[i];
        const int end = RowPtrA[i + 1];
        for (int colidx = start; colidx < end; colidx++) {
            // const int j = csrColIndA[colidx] - baseA; 
            double A_abs = fabs(ValA[colidx]);
            sum += A_abs;
        }
        norminf = (norminf > sum) ? norminf : sum;
    }
    return norminf;
}


void testFidesys(const int colsA, const double* h_X, const double* fid_X, FILE *log)
{
    FILE* DIFF;
    DIFF = fopen("../output/diff.txt", "w");
    double* h_difference;
    h_difference = (double*)malloc(sizeof(double) * colsA);
    for (int i = 0; i < colsA; ++i) {
        h_difference[i] = h_X[i] - fid_X[i];
        fprintf(DIFF, "%e\n", h_difference[i]);
    }
    fclose(DIFF);
    DIFF = NULL;
    fprintf(log, "inf-norm of |Fid(X)-CU(X)| =  %e\n", vec_norminf(colsA, h_difference));
}



#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}

#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysctl.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif