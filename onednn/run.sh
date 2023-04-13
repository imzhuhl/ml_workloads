set -aux;

cmake -S . -B build && cmake --build build -j 4 && ./build/test_sgemm
