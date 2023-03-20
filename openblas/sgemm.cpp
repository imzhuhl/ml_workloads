#include <chrono>
#include <cstring>
#include <iostream>

#include "cblas.h"
#include "utils.hpp"

constexpr int REP_CNT = 10;

int native_c(int M, int K, int N, float *A, float *B, float *C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = 0.0;
      for (int p = 0; p < K; p++) {
        tmp += A[p * M + i] * B[p + j * K];
      }
      C[j * M + i] += tmp;
    }
  }
  return 0;
}

int run_sgemm(int *mat_size, bool verbose) {
  int M, N, K;
  if (mat_size[0] != 0 && mat_size[1] == 0 && mat_size[2] == 0) {
    M = mat_size[0];
    N = M;
    K = M;
  } else if (mat_size[0] != 0 && mat_size[1] != 0 && mat_size[2] != 0) {
    M = mat_size[0];
    N = mat_size[1];
    K = mat_size[2];
  }

  double gflops = 2.0 * M * N * K * 1.0e-09;

  float alpha = 1;
  float beta = 0;
  int lda = M;  // col major order
  int ldb = K;
  int ldc = M;

  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  float *myc = new float[M * N];

  fill_array(A, M * K, InitVecFlag::IncreaseByOne);
  fill_array(B, K * N, InitVecFlag::IncreaseByOne);
  fill_array(C, M * N, InitVecFlag::Zero);

  /* Time of implementation */
  double time_best = 99999;
  for (int rep = 0; rep < REP_CNT; rep++) {
    copy_array(C, myc, M * N);
    auto start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
                lda, B, ldb, beta, myc, ldc);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("%.2lf GFLOPS, %.2lf ms\n", gflops / (elapsed.count() * 1.0e-3),
           elapsed.count());
    time_best = std::min(time_best, elapsed.count() * 1.0e-3);
  }
  printf("TARGET: %.2lf GFLOPS\n", gflops / time_best);

  // native_c(M, K, N, A, B, C);
  // compare_array(C, myc, M * N);

  if (verbose) {
    printf("Matrix A:\n");
    display_matrix(A, M, K);
    printf("Matrix B:\n");
    display_matrix(B, K, N);
    printf("Matrix C:\n");
    display_matrix(C, M, N);
    printf("Matrix myc:\n");
    display_matrix(myc, M, N);
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] myc;

  return 0;
}

int main(int argc, char **argv) {
  printf("argc: %d\n", argc);
  if (argc <= 1 || argc > 5) {
    printf("Please run:\n1./sbgemm mat_size [-v]\n./sbgemm m n k [-v]");
    return 0;
  }
  bool verbose = false;
  int mat_size[3] = {0};
  int idx = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp("-v", argv[i]) == 0) {
      verbose = true;
    } else {
      mat_size[idx++] = atoi(argv[i]);
    }
  }

  run_sgemm(mat_size, verbose);
  return 0;
}
