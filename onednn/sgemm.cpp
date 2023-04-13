#include <iostream>
#include "oneapi/dnnl/dnnl.hpp"


int main() {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    printf("Hello World\n");
    return 0;
}

