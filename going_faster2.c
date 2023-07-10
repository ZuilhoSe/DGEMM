#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dgemm (int n, double* A, double* B, double* C){
    int i, j, k;
    for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j){
        double cij = C[i+j*n]; /* cij = C[i][j] */
        for(k = 0; k < n; k++ )
            cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
            C[i+j*n] = cij; /* C[i][j] = cij */

    }
}

void randomize_matrix(double *A, int m, int n){
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            A[i * n + j] = (double)(rand() % 100) + 0.01 * (rand() % 100);
            if (rand() % 2 == 0) A[i * n + j] *= 1.0;
        }
    }
}

int main(){
    printf("Creating Matrix\n");
    size_t n = 4096;
    size_t i,j;
    clock_t start, end;

    double *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;

    A=(double *)malloc(sizeof(double)*n*n);
    B=(double *)malloc(sizeof(double)*n*n);
    C=(double *)malloc(sizeof(double)*n*n);

    randomize_matrix(A,n,n);
    randomize_matrix(B,n,n);
    randomize_matrix(C,n,n);

    printf("Starting\n");
    clock_t start_time = clock();
    dgemm(n, A, B, C);
    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.4f seconds\n", elapsed_time);

    free(A);
    free(B);
    free(C);

    printf("done\n");
    return 0;
}

// compile code 
// gcc -O3 -mavx -o executables/going_faster2 going_faster2.c