#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "utils.h"

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

int run_sgemm(int M, int N, int K, bool verbose) {
  double num_gflop = 2.0 * M * N * K * 1.0e-09;

  float alpha = 1;
  float beta = 0;
  int lda = M; // col major order
  int ldb = K;
  int ldc = M;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);
  std::vector<float> dst_C(M * N);

  fill_array(A, InitValFlag::IncreaseByOne);
  fill_array(B, InitValFlag::IncreaseByOne);
  fill_array(C, InitValFlag::Zero);

  /* Time of implementation */
  double best_gflops = 0;
  for (int rep = 0; rep < REP_CNT; rep++) {
    copy_array(C, dst_C);
    auto start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                A.data(), lda, B.data(), ldb, beta, dst_C.data(), ldc);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double cur_gflops = num_gflop / (elapsed.count() * 1.0e-3);
    printf("%.2lf GFLOPS, %.2lf ms\n", cur_gflops, elapsed.count());

    best_gflops = cur_gflops > best_gflops ? cur_gflops : best_gflops;
  }
  printf("TARGET: %.2lf GFLOPS\n", best_gflops);

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
    display_matrix(dst_C, M, N);
  }

  return 0;
}

void bad_args() {
  std::cerr << "Usage: sgemm m n k [-v]\n"
               "       sgemm size [-v]\n"
               "If a single <size> is specified, it is used for all three "
               "dimensions (m/n/k).\n";
  throw std::invalid_argument("Incorrect input arguments.");
}

int main(int argc, char **argv) {
  printf("argc: %d\n", argc);

  bool verbose = false;
  int m, n, k;
  if (strcmp("-v", argv[argc - 1]) == 0) {
    verbose = true;
    argc--;
  }
  if (argc == 2) {
    m = n = k = std::atoi(argv[1]);
  } else if (argc == 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  } else {
    bad_args();
  }

  printf("MatMul: m, n, k: %d, %d, %d\n", m, n, k);

  run_sgemm(m, n, k, verbose);
  return 0;
}
