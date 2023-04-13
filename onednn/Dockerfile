FROM ubuntu:22.04

RUN sed -i -E 's/(security|archive|ports).ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt update && apt install -y build-essential cmake git


# Build oneDNN
RUN git clone 
WORKDIR oneDNN
# RUN mkdir -p build && cd build
# RUN cmake .. -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DCMAKE_BUILD_TYPE=Debug -DONEDNN_BUILD_GRAPH=OFF
# RUN cmake .. -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_BUILD_GRAPH=OFF -DCMAKE_INSTALL_PREFIX=/hostfs/opt
# RUN make -j32

RUN cmake -S . -B build -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_BUILD_GRAPH=OFF -DCMAKE_INSTALL_PREFIX=/opt/oneDNN
RUN cmake --build build -j 32 --target install
