# quay.io/pypa/manylinux2014_x86_64:latest
# quay.io/pypa/manylinux2014_aarch64:latest

set -e

# x86
yum install -y devtoolset-11-gcc-c++ devtoolset-11-gcc-gfrotran
scl enable devtoolset-11 bash

yum install -y ninja-build
export PATH=/opt/_internal/cpython-3.10.11/bin:$PATH

pip config set global.index-url https://pypi.douban.com/simple/
pip install --no-cache-dir astunparse pyyaml typing_extensions numpy cffi \
    setuptools future six requests dataclasses filelock jinja2 networkx sympy

git clone -b spr_sbgemm_fix https://github.com/imzhuhl/OpenBLAS.git --depth=1
cd OpenBLAS && BUILD_BFLOAT16=1 DYNAMIC_ARCH=1 USE_OPENMP=1 make -j32 && make PREFIX=/opt/OpenBLAS install

# assume there is a pytorch repo
cd pytorch

PYTORCH_BUILD_VERSION=2.1.0+openblas PYTORCH_BUILD_NUMBER=1 \
MAX_JOBS=32 USE_OPENMP=1 USE_DISTRIBUTED=0 BUILD_TEST=0 USE_CUDA=0 USE_FBGEMM=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_MKLDNN=0 \
python setup.py bdist_wheel



