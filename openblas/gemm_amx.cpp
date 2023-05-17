#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

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

void native_c() {}

void init_array(std::vector<int8_t>& a, int8_t max_val) {
  std::fill(a.begin(), a.end(), 0);

  int8_t val = 0;
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = i;
    if (i >= max_val) return;
  }
}

template <typename T>
void print_array(std::vector<T>& a, int m, int n,
                 std::vector<int32_t> stride) {
  // stride: [x, y]
  // if row major matrix, stride is [colnum, 1]
  // else col major, stride is [1, rownum]

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf(" %d", a[i * stride[0] + j * stride[1]]);
    }
    printf("\n");
  }
}

int main() {
  if (!support_amx_bf16()) return 0;

  set_tiledata_use();

  int M = 16;
  int K = 64;
  int N = 16;

  std::vector<int8_t> A(M * K, 0);
  std::vector<int8_t> B(N * K, 0);
  std::vector<int32_t> C(M * N, 0);

  tilecfg cfg;
  memset(&cfg, 0, sizeof(tilecfg));
  cfg.palette_id = 1;
  cfg.tile_colsb[0] = 64;
  cfg.tile_colsb[1] = 64;
  cfg.tile_colsb[2] = 64;
  cfg.tile_rows[0] = 16;
  cfg.tile_rows[1] = 16;
  cfg.tile_rows[2] = 16;
  _tile_loadconfig(&cfg);

  init_array(A, 127);
  init_array(B, 127);
  printf("A:\n");
  print_array(A, M, K, {K, 1});
  printf("B:\n");
  print_array(B, N, K, {K, 1});


  _tile_loadd(1, A.data(), 64);
  _tile_loadd(2, B.data(), 64);
  _tile_loadd(0, C.data(), 64);

  _tile_dpbssd(0, 1, 2);

  _tile_stored(0, C.data(), 64);

  printf("C:\n");
  print_array(C, M, N, {N, 1});

  return 0;
}
