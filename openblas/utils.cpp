#include "utils.h"
#include "openblas_config.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>
#include <sys/syscall.h>
#include <unistd.h>

void compare_array(std::vector<float> &a, std::vector<float> &b) {
  assert(a.size() == b.size());

  float diff = 0.0;
  for (int i = 0; i < a.size(); i++) {
    diff = std::abs(a[i] - b[i]);
    if (diff > 0.5) {
      printf("Check error: %.3f vs %.3f\n", a[i], b[i]);
      return;
    }
  }
  printf("Check pass.\n");
  return;
}

void fill_array(std::vector<float> &v, InitValFlag flag) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < v.size(); i++) {
    switch (flag) {
      case InitValFlag::Zero:
        v[i] = 0;
        break;
      case InitValFlag::One:
        v[i] = 1;
        break;
      case InitValFlag::IncreaseByOne:
        v[i] = i;
        break;
      case InitValFlag::RandonValue:
        v[i] = dist(mt);
        break;
      default:
        printf("Error InitValFlag value\n");
        exit(1);
    }
  }
}

void copy_array(std::vector<float> &src, std::vector<float> &dst) {
  assert(src.size() == dst.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i];
  }
}

void display_matrix(std::vector<float> &mat, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf(" %7.3f", mat[j * M + i]);
    }
    printf("\n");
  }
  printf("\n");
}

bfloat16 fp32_to_bf16(float src) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&src);
  uint16_t b = (*a) >> 16;
  return *(reinterpret_cast<bfloat16 *>(&b));
}

bfloat16 round_fp32_to_bf16(float src) {
  uint32_t *fromptr = reinterpret_cast<uint32_t *>(&src);
  uint16_t res = (*fromptr >> 16);
  const uint16_t error = (*fromptr & 0x0000ffff);
  uint16_t bf_l = res & 0x0001;
  if ((error > 0x8000) || ((error == 0x8000) && (bf_l != 0))) {
    res += 1;
  }
  return *(reinterpret_cast<bfloat16 *>(&res));
}

float bf16_to_fp32(bfloat16 src) {
  uint16_t *a = reinterpret_cast<bfloat16 *>(&src);
  uint32_t b = *a;
  b <<= 16;
  return *(reinterpret_cast<float *>(&b));

}

void array_fp32_to_bf16(std::vector<float> &src, std::vector<bfloat16> &dst) {
  assert(src.size() == dst.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = fp32_to_bf16(src[i]);
  }
}

void array_bf16_to_fp32(std::vector<bfloat16> &src, std::vector<float> &dst) {
  assert(src.size() == dst.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = bf16_to_fp32(src[i]);
  }
}

/**
 * Invoke a Linux system call to request access to Intel® AMX features.
 * So we detect architecture.
*/

#if defined(__x86_64__)
void cpuid(int op, int *eax, int *ebx, int *ecx, int *edx) {
  __asm__ __volatile__
    ("cpuid": "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) : "a" (op) , "c" (0) : "cc");
}

void cpuid_count(int op, int count ,int *eax, int *ebx, int *ecx, int *edx) {
  __asm__ __volatile__
    ("cpuid": "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) : "0" (op), "2" (count) : "cc");
}

int support_amx_bf16() {
  int eax, ebx, ecx, edx;
  int ret=0;

  // CPUID.7.0:EDX indicates AMX support
  cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
  if ((edx & BIT_AMX_TILE) && (edx & BIT_AMX_BF16)) {
    // CPUID.D.0:EAX[17:18] indicates AMX enabled
    cpuid_count(0xd, 0, &eax, &ebx, &ecx, &edx);
    if ((eax & BIT_AMX_ENBD) == BIT_AMX_ENBD)
      ret = 1;
  }
  return ret;
}

bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
    return false;
  }
  return true;
}

#endif  // __x86_64__
