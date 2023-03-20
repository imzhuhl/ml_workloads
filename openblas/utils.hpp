#include <cstddef>
#include <iostream>
#include <random>


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
    if (diff > 1e-3) {
      printf("Check error: %.2f vs %.2f\n", a[i], b[i]);
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
