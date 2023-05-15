project(OpenBLAS-Test C CXX)

set(OPENBLAS_DIR "/opt/OpenBLAS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(sgemm sgemm.cpp utils.cpp)
target_include_directories(sgemm PUBLIC ${OPENBLAS_DIR}/include)
target_link_directories(sgemm PUBLIC ${OPENBLAS_DIR}/lib)
target_link_libraries(sgemm openblas)


add_executable(test_amxtile test_amxtile.cpp utils.cpp)
target_include_directories(test_amxtile PUBLIC ${OPENBLAS_DIR}/include)
target_link_directories(test_amxtile PUBLIC ${OPENBLAS_DIR}/lib)
target_link_libraries(test_amxtile openblas)
