project(oneDNN-Test C CXX)

set(ROOT_DIR ${CMAKE_SOURCE_DIR})
set(ONEDNN_DIR "/hostfs/opt/oneDNN")

message(STATUS "ROOT_DIR : " ${ROOT_DIR})

add_executable(test_sgemm test_sgemm.cpp)
target_include_directories(test_sgemm)
