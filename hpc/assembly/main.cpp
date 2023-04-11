#include <iostream>
#include <cstdint>

extern "C" {
  void load_asm(uint64_t const * i_a);
}


int main() {
    uint64_t * lst = new uint64_t[10];

    for(size_t i = 0; i < 10; i++ ) {
        lst[i] = (i+1) * 16;
    }

    load_asm(lst + 2);

    delete[] lst;

    return 0;
}

