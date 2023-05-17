cmake_minimum_required(VERSION 3.18)

project(OpenBLAS-Test C CXX)

set(OPENBLAS_DIR "/opt/OpenBLAS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include_directories(${OPENBLAS_DIR}/include)
link_directories(${OPENBLAS_DIR}/lib)
link_libraries(openblas)

add_executable(sgemm sgemm.cpp utils.cpp)

add_executable(sbgemm sbgemm.cpp utils.cpp)

add_executable(compare_sgemm_sbgemm compare_sgemm_sbgemm.cpp utils.cpp)

add_executable(test_amxtile test_amxtile.cpp utils.cpp)
