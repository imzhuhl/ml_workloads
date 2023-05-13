project(OpenBLAS-Test C CXX)

set(OPENBLAS_DIR "/opt/OpenBLAS")

add_executable(sgemm sgemm.cpp utils.cpp)
target_include_directories(sgemm PUBLIC ${OPENBLAS_DIR}/include)
target_link_directories(sgemm PUBLIC ${OPENBLAS_DIR}/lib)
target_link_libraries(sgemm openblas)
