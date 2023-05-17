#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "openblas_config.h"
#include "utils.h"

// Define tile config data structure
// format of memory payload. each field is a byte.
// 0: palette_id
// 1: startRow (8b)
// 2-15: reserved (must be zero)
// 16-17: tile0.colsb -- bytes_per_row
// 18-19: tile1.colsb
// 20-21: tile2.colsb
// ...
// 46-47: tile15.colsb
// 48: tile0.rows
// 49: tile1.rows
// 50: tile2.rows
// ...
// 63: tile15.rows
struct tilecfg {
  char palette_id;
  char start_row;
  char dummy0[14];  // bytes 2-15 reserved, must be zero
  short tile_colsb[8];
  char dummy1[16];  // bytes 32-47 reserved, must be zero
  char tile_rows[8];
  char dummy2[16];  // bytes 56-63 reserved, must be zero
};

int main() {
  if (!support_amx_bf16()) return 0;

  set_tiledata_use();

  tilecfg cfg;
  memset(&cfg, 0, sizeof(tilecfg));
  cfg.palette_id = 1;
  cfg.tile_colsb[0] = 64;
  cfg.tile_rows[0] = 16;
  _tile_loadconfig(&cfg);

  int M = 16;
  int N = 32;

  std::vector<float> origin(M * N, 0);
  std::vector<bfloat16> A(M * N, 0);
  std::vector<bfloat16> B(M * N, 0);
  std::vector<float> FA(M * N, 0);
  std::vector<float> FB(M * N, 0);

  fill_array(origin, InitValFlag::IncreaseByOne);
  array_fp32_to_bf16(origin, A);

  _tile_loadd(0, A.data(), 64);
  _tile_stored(0, B.data(), 64);

  array_bf16_to_fp32(A, FA);
  array_bf16_to_fp32(B, FB);
  display_matrix(FA, M, N);
  display_matrix(FB, M, N);

  return 0;
}
