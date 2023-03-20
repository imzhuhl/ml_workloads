#include <arm_neon.h>
#include <chrono>
#include <iostream>
#include <cstddef>
#include <algorithm>
#include <random>


void fmla_kernel(int cnt) {
    __asm__ __volatile__(
        "mov x0, %[cnt]\n"
        "1:\n"
        "fmla v0.4s, v1.4s, v2.4s\n"
        "fmla v3.4s, v4.4s, v5.4s\n"
        "fmla v6.4s, v7.4s, v8.4s\n"
        "fmla v9.4s, v10.4s, v11.4s\n"
        // "fmla v12.4s, v13.4s, v14.4s\n"
        // "fmla v15.4s, v16.4s, v17.4s\n"
        "subs x0, x0, #1\n"
        "bne 1b\n"
        :
        : [cnt] "r" (cnt)
        : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5",
          "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17"
    );
}

void fill_random(float* v, size_t length) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
}


void run_fmla_native() {
    int rep = 10000;
    int flop_per_fmla = 8;  // 4 mul + 4 add
    int fmla_inst_num = 4;
    double flop = rep * flop_per_fmla * fmla_inst_num;
    auto st = std::chrono::steady_clock::now();
    fmla_kernel(rep);
    auto et = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = et - st;
    std::cout << "Time(ms): " << elapsed.count() << std::endl;
    std::cout << "GFLOPS: " << (flop * 1.0e-9) / (elapsed.count() / 1000) << std::endl;
}

int main() {
    run_fmla_native();
    return 0;
}
