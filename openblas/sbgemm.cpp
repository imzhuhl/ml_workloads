#include <chrono>
#include <cstring>
#include <iostream>

#include "cblas.h"
#include "utils.hpp"

inline uint16_t float_to_bf16(const float v) {
  const uint32_t *fromptr = reinterpret_cast<const uint32_t *>(&v);
#if defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC)
  uint16_t res;

  __asm __volatile(
      "ldr    s0, [%[fromptr]]\n"
      ".inst  0x1e634000\n"  // BFCVT h0, s0
      "str    h0, [%[toptr]]\n"
      :
      : [fromptr] "r"(fromptr), [toptr] "r"(&res)
      : "v0", "memory");
#else
  uint16_t res = (*fromptr >> 16);
  const uint16_t error = (*fromptr & 0x0000ffff);
  uint16_t bf_l = res & 0x0001;
  if ((error > 0x8000) || ((error == 0x8000) && (bf_l != 0))) {
    res += 1;
  }
#endif  // __ARM_FEATURE_BF16_SCALAR_ARITHMETIC
  return res;
}

void vec_fp32_to_bf16(float *src, uint16_t *dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = float_to_bf16(src[i]);
  }
}

int run_sbgemm(int *mat_size, bool verbose) {
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

  float *FA = new float[M * K];
  float *FB = new float[K * N];
  float *C = new float[M * N];
  float *myc = new float[M * N];
  fill_array(FA, M * K, InitVecFlag::RandonValue);
  fill_array(FB, K * N, InitVecFlag::RandonValue);
  fill_array(C, M * N, InitVecFlag::Zero);

  // transpose(FA, M, K);
  // transpose(FB, K, N);

  bfloat16 *A = new bfloat16[M * K];
  bfloat16 *B = new bfloat16[K * N];
  vec_fp32_to_bf16(FA, (uint16_t *)A, M * K);
  vec_fp32_to_bf16(FB, (uint16_t *)B, K * N);

  double best_GFOPS = 0.0;
  for (int rep = 0; rep < 10; rep++) {
    copy_array(C, myc, M * N);
    auto start = std::chrono::steady_clock::now();
    cblas_sbgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
                 lda, B, ldb, beta, myc, ldc);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double dtime = elapsed.count() * 1.0e-3;  // s
    printf("%.2lf GFLOPS, %.2lf ms\n", gflops / dtime, elapsed.count());
    best_GFOPS = gflops / dtime;
  }

  printf("TARGET: %.2lf GFLOP/S\n", best_GFOPS);

  // sbgemm_native_c(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  // compare_matrix(C, myc, M, N);

  if (verbose) {
    printf("Matrix A:\n");
    display_matrix(FA, M, K);
    printf("Matrix B:\n");
    display_matrix(FB, K, N);
    printf("Matrix C:\n");
    display_matrix(C, M, N);
    printf("Matrix myc:\n");
    display_matrix(myc, M, N);
  }

  delete[] FA;
  delete[] FB;
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] myc;

  return 0;
}

int main(int argc, char **argv) {
#if defined (__x86_64__)
  if (support_amx_bf16()) {
    set_tiledata_use();
  }
#endif
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

  run_sbgemm(mat_size, verbose);
  return 0;
}
