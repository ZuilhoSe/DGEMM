#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UNROLL (4)
#define BLOCKSIZE 32

void do_block (int n, int si, int sj, int sk, double *A, double *B, double *C){
    for ( int i = si; i < si+BLOCKSIZE; i+=UNROLL*8 )
        for ( int j = sj; j < sj+BLOCKSIZE; j++) {
        __m256d c[UNROLL];
        for ( int r = 0; r < UNROLL; r++)
            c[r] = _mm256_load_pd(C+i+r*4+j*n);
        for (int k = sk; k < sk+BLOCKSIZE; k++ ){
            __m256d b = _mm256_broadcast_sd(B+k+j*n);
            for (int r = 0; r < UNROLL; r++)
                c[r] = _mm256_add_pd(c[r], _mm256_mul_pd(_mm256_load_pd(A+n*k+r*4+i), b));
        }
        for ( int r = 0; r < UNROLL; r++ )
            _mm256_store_pd(C+i+r*4+j*n, c[r]);
        }
} 

void dgemm (int n, double* A, double* B, double* C){
    for(int sj = 0; sj < n; sj += BLOCKSIZE) 
        for(int si = 0; si < n; si += BLOCKSIZE) 
            for(int sk = 0; sk < n; sk += BLOCKSIZE) 
                do_block(n, si, sj, sk, A, B, C); 
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

void copy_matrix(double *src, double *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++){
        *(dest + i) = *(src + i);
    }
    if (i != n){
        printf("copy failed at %d while there are %d elements in total.\n", i, n);
    }
}

int main() {

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

    // Cleanup
    free(A);
    free(B);
    free(C);

    printf("done\n");
    return 0;
}

// compile code 
// gcc -O3 -mavx -o executables/going_faster5 going_faster5.c