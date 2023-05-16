#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "utils.h"

void vec_fp32_to_bf16(float *src, bfloat16 *dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    dst[i] = *(reinterpret_cast<bfloat16 *>(&src[i]));
#else
    dst[i] = *(reinterpret_cast<bfloat16 *>(&src[i]) + 1);
#endif
  }
}


void debug_fill_array(std::vector<float> &v) {
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = 1;
  }
}


void compare(int *mat_size, bool verbose) {
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
  float alpha = 1;
  float beta = 0;
  int lda = M;  // col major order
  int ldb = K;
  int ldc = M;

  std::vector<float> FA(M * K, 0);
  std::vector<float> FB(K * N, 0);
  std::vector<float> sgemm_C(M * N, 0);
  std::vector<float> sbgemm_C(M * N, 0);

  // fill_array(FA, InitValFlag::RandonValue);
  // fill_array(FB, InitValFlag::RandonValue);

  // debug fill array
  for (size_t i = 0; i < FA.size(); i++) {
    FA[i] = 1;
  }
  for (size_t i = 0; i < FB.size(); i++) {
    FB[i] = i % 256;
  }
  //

  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  array_fp32_to_bf16(FA, A);
  array_fp32_to_bf16(FB, B);

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              FA.data(), lda, FB.data(), ldb, beta, sgemm_C.data(), ldc);
  cblas_sbgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
               A.data(), lda, B.data(), ldb, beta, sbgemm_C.data(), ldc);

  if (verbose) {
    printf("Matrix A:\n");
    display_matrix(FA, M, K);
    printf("Matrix B:\n");
    display_matrix(FB, K, N);
    printf("Matrix sgemm_C:\n");
    display_matrix(sgemm_C, M, N);
    printf("Matrix sbgemm_C:\n");
    display_matrix(sbgemm_C, M, N);
  }

  compare_array(sgemm_C, sbgemm_C);
  
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

  compare(mat_size, verbose);
  return 0;
}
