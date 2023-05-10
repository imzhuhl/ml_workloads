FROM ubuntu:22.04

RUN sed -i -E 's/(security|archive|ports).ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

WORKDIR OpenBLAS
# RUN DYNAMIC_ARCH=1 USE_OPENMP=1 make -j32 && make PREFIX=/opt/OpenBLAS install
RUN BUILD_BFLOAT16=1 USE_OPENMP=1 make -j32 && make install
