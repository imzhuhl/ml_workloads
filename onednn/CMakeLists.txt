project(oneDNN-Test C CXX)

set(ROOT_DIR ${CMAKE_SOURCE_DIR})
set(ONEDNN_DIR "/opt/oneDNN")

message(STATUS "ROOT_DIR : " ${ROOT_DIR})

add_executable(sgemm sgemm.cpp)

target_include_directories(sgemm PUBLIC ${ONEDNN_DIR}/include)
target_link_directories(sgemm PUBLIC ${ONEDNN_DIR}/lib)
target_link_libraries(sgemm dnnl)
