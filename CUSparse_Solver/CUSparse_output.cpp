#include "CUSparse_helper.h"

void denseVectorFileOutput(char const* FileName, int rowsA, const double* h_ValA) {
    FILE* inA;
    inA = fopen(FileName, "w");
    for (int row = 0; row < rowsA; row++) {
        fprintf(inA, "%e\n", h_ValA[row]);
    }
    fclose(inA);
    inA = NULL;
}