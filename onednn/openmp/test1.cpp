#include <iostream>
#include <omp.h>


int main() {

    #pragma omp parallel num_threads(4)
    {
        int nthr_ = omp_get_num_threads();
        int ithr_ = omp_get_thread_num();
        printf("Hello World: %d, %d\n", nthr_, ithr_);
    }

    return 0;
}
