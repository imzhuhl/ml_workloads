set -aux;

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DCMAKE_BUILD_TYPE=Debug -S . -B build \
&& cmake --build build -j 4


# ./build/sgemm 100
# ONEDNN_DEFAULT_FPMATH_MODE=BF16 ./build/sgemm 100
