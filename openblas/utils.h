#pragma once

#include <vector>
#include <cstdint>

#include "cblas.h"
#include "openblas_config.h"

enum class InitValFlag {
  Zero,
  One,
  IncreaseByOne,
  RandonValue,
};

void compare_array(std::vector<float> &a, std::vector<float> &b);

void fill_array(std::vector<float> &v, InitValFlag flag);

void copy_array(std::vector<float> &src, std::vector<float> &dst);

void display_matrix(std::vector<float> &mat, int M, int N);

void array_fp32_to_bf16(std::vector<float> &src, std::vector<bfloat16> &dst);

void array_bf16_to_fp32(std::vector<bfloat16> &src, std::vector<float> &dst);

bfloat16 fp32_to_bf16(float src);

bfloat16 round_fp32_to_bf16(float src);

float bf16_to_fp32(bfloat16 src);

#if defined(__x86_64__)

void cpuid(int op, int *eax, int *ebx, int *ecx, int *edx);
void cpuid_count(int op, int count ,int *eax, int *ebx, int *ecx, int *edx);

#define BIT_AMX_TILE 0x01000000
#define BIT_AMX_BF16 0x00400000
#define BIT_AMX_ENBD 0x00060000
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

int support_amx_bf16();
bool set_tiledata_use();

#endif  // __x86_64__
