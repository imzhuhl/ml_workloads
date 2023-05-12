#include <immintrin.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>

#include "utils.hpp"


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


void native_c() {

}


void init_array(std::vector<int8_t> v, int8_t max_val) {
  int8_t val = 0;
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = val++;
    if (val > max_val) val -= max_val;
  }
}

void print_array(std::vector<int8_t> v, int m, int n, int stride) {

}


int main() {
  if (!support_amx_bf16()) return 0;

  set_tiledata_use();

  int M = 32;
  int K = 32;
  int N = 32;

  std::vector<int8_t> A(M * K, 0);
  std::vector<int8_t> B(K * N, 0);
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


  _tile_loadd(1, A.data(), 64);
  _tile_loadd(2, B.data(), 64);
  _tile_loadd(0, C.data(), 64);

  _tile_dpbssd (0, 1, 2);

  _tile_stored(0, C.data(), 64);

  return 0;
}
