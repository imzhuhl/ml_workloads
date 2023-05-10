#include <random>
#include <vector>

enum class InitValFlag {
  Zero,
  One,
  IncreaseByOne,
  RandonValue,
};

void fill_array(std::vector<float> v, InitValFlag flag) {
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