#include "CUSparse_helper.h"

void csrMatrixFileInput(char const* FileName, int* rowsA, int* nnzA,int** h_RowPtrA, int** h_ColIndA, double** h_ValA) {
	FILE* inA;
	inA = fopen(FileName, "r");
    if (inA == NULL) {
        printf("Can't read matrix");
        exit(EXIT_FAILURE);
    }
	fscanf(inA, "%d", rowsA);
	fscanf(inA, "%d", nnzA);
    
	*h_RowPtrA = (int*)malloc(sizeof(int) * (*rowsA + 1));
	*h_ColIndA = (int*)malloc(sizeof(int) * (*nnzA));
	*h_ValA = (double*)malloc(sizeof(double) * (*nnzA));
    
    for (int i = 0; i < *rowsA + 1; ++i) {
        fscanf(inA, "%d", *h_RowPtrA+i);
    }
    for (int i = 0; i < *nnzA; ++i) {
        fscanf(inA, "%d", *h_ColIndA+i);
    }
    for (int i = 0; i < *nnzA; ++i) {
        fscanf(inA, "%lf", *h_ValA+i);
    }
    fclose(inA);
    inA = NULL;
}

void denseVectorFileInput(char const* FileName, int* rowsA, double** h_ValA) {
    FILE* inA;
    inA = fopen(FileName, "r");
    if (inA == NULL) {
        printf("Can't read vector");
        exit(EXIT_FAILURE);
    }
    fscanf(inA, "%d", rowsA);
    *h_ValA = (double*)malloc(sizeof(double) * (*rowsA));
    for (int row = 0; row < *rowsA; row++) {
        fscanf(inA, "%lf", *h_ValA+row);
    }
    fclose(inA);
    inA = NULL;

}