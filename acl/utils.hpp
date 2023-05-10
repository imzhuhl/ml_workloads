#pragma once

#include <iostream>
#include <string>
#include <cassert>
#include <random>
#include <cmath>
#include <cstddef>

enum class MAT_ORDER {
    ColMajor,
    RowMajor,
};

enum class InitVecFlag {
    Zero,
    One,
    IncreaseByOne,
    RandonValue,
};


void copy_array(float *src, float *dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

void fill_vector(float *v, size_t length, InitVecFlag flag) {
    std::random_device rd;
    std::mt19937 mt(12);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (size_t i = 0; i < length; i++) {
        switch (flag)
        {
        case InitVecFlag::Zero : 
            v[i] = 0;
            break;
        case InitVecFlag::One : 
            v[i] = 1;
            break;
        case InitVecFlag::IncreaseByOne : 
            v[i] = i;
            break;
        case InitVecFlag::RandonValue :
            v[i] = dist(mt);
            break;
        default:
            printf("Error InitVecFlag value\n");
            exit(1);
        }
    }
}

void display_matrix(float *v, size_t length, size_t ldv, MAT_ORDER order) {
    assert(length > 0 && ldv > 0);
    assert(length % ldv == 0);
    assert(length >= ldv);
    if (order == MAT_ORDER::ColMajor) {
        size_t col_num = length / ldv;
        size_t row_num = ldv;
        for (size_t i = 0; i < row_num; i++) {
            for (size_t j = 0; j < col_num; j++) {
                printf(" %7.3f", v[j*ldv+i]);
            }
            printf("\n");
        }
        printf("\n");
    } else if (order == MAT_ORDER::RowMajor) {
        size_t row_num = length / ldv;
        size_t col_num = ldv;
        for (size_t i = 0; i < row_num; i++) {
            for (size_t j = 0; j < col_num; j++) {
                printf(" %7.3f", v[i*ldv+j]);
            }
            printf("\n");
        }
        printf("\n");
    } else {
        std::cout << "Error MatOrder value" << std::endl;
        exit(1);
    }
}

void compare_array(float *a, float *b, size_t size) {
    float diff = 0.0;
    for (int i = 0; i < size; i++) {
        diff = std::abs(a[i] - b[i]);
        if (diff > 1e-3) {
            printf("Check error: %.2f vs %.2f\n", a[i], b[i]);
            return;
        }
    }
    printf("Check pass.\n");
    return;
}

void compare_matrix(float *a, float *b, int m, int n) {
  float diff = 0.0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      diff = std::abs(a[j * m + i] - b[j * m + i]);
      if (diff > 1.0) {
        printf("Check error %.5f vs %.5f, at (%d,%d)\n", a[j * m + i], b[j * m + i], i, j);
        return;
      }
    }
  }
  printf("Check pass.\n");
  return;
}
