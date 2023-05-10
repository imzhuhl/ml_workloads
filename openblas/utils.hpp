#include <cstddef>
#include <iostream>
#include <random>
#include <sys/syscall.h>
#include <unistd.h>

enum class InitVecFlag {
  Zero,
  One,
  IncreaseByOne,
  RandonValue,
};

void compare_array(float *a, float *b, int size) {
  float diff = 0.0;
  for (int i = 0; i < size; i++) {
    diff = std::abs(a[i] - b[i]);
    if (diff > 1e-2) {
      printf("Check error: %.3f vs %.3f\n", a[i], b[i]);
      return;
    }
  }
  printf("Check pass.\n");
  return;
}

void fill_array(float *v, size_t length, InitVecFlag flag) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < length; i++) {
    switch (flag) {
      case InitVecFlag::Zero:
        v[i] = 0;
        break;
      case InitVecFlag::One:
        v[i] = 1;
        break;
      case InitVecFlag::IncreaseByOne:
        v[i] = i;
        break;
      case InitVecFlag::RandonValue:
        v[i] = dist(mt);
        break;
      default:
        printf("Error InitVecFlag value\n");
        exit(1);
    }
  }
}

void copy_array(float *src, float *dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

void display_matrix(float *mat, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf(" %7.3f", mat[j * M + i]);
    }
    printf("\n");
  }
  printf("\n");
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

#define BIT_AMX_TILE 0x01000000
#define BIT_AMX_BF16 0x00400000
#define BIT_AMX_ENBD 0x00060000
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

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
