#include <immintrin.h>

#include <iostream>
#include <vector>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64

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
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;


int main() {
  std::vector<uint16_t> data(1024);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  for (auto& v : data) {
    printf(" %d", v);
  }
  printf("\n");

  uint16_t* pa = data.data();

  __tilecfg cfg;
  cfg.palette_id = 1;
  cfg.start_row = 0;
  cfg.colsb[0] = MAX_ROWS;
  cfg.rows[0] = MAX_ROWS;
  for (int i = 1; i < 4; ++i){
    cfg.colsb[i] = MAX_COLS;
    cfg.rows[i] =  MAX_ROWS;
  }
  _tile_loadconfig(&cfg);

  _tile_loadd(0, pa, 2);

  return 0;
}