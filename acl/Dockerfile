FROM ubuntu:22.04

WORKDIR ComputeLibrary
RUN git checkout v22.11

# RUN scons -j32 Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
#   os=linux arch=armv8.2-a build=native multi_isa=1 \
#   experimental_fixed_format_kernels=1 openmp=1 cppthreads=0 \
#   build_dir=/opt/ComputeLibrary/build

RUN scons -j32 Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
    os=linux arch=armv8.6-a-sve2 build=native \
    experimental_fixed_format_kernels=1 openmp=1 cppthreads=0 \
    install_dir=/opt/ComputeLibrary
