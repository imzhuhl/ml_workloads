OPENBLAS_INCLUDE=/opt/OpenBLAS/include
OPENBLAS_LIB=/opt/OpenBLAS/lib

COMPILE_OPTIONS=-I$(OPENBLAS_INCLUDE) -L$(OPENBLAS_LIB) -Wl,-rpath,$(OPENBLAS_LIB) -lopenblas

sgemm: sgemm.cpp utils.hpp
	g++ -O2 sgemm.cpp $(COMPILE_OPTIONS) -o sgemm

sbgemm: sbgemm.cpp utils.hpp
	g++ -O2 sbgemm.cpp $(COMPILE_OPTIONS) -o sbgemm

compare_sgemm_sbgemm: compare_sgemm_sbgemm.cpp utils.hpp
	g++ -g compare_sgemm_sbgemm.cpp $(COMPILE_OPTIONS) -o compare_sgemm_sbgemm

amx_example: amx_example.cpp
	g++ -O2 -march=native amx_example.cpp -o amx_example

gemm_amx: gemm_amx.cpp
	g++ -g -march=native gemm_amx.cpp -o gemm_amx

clean:
	rm -f sbgemm sgemm compare_sgemm_sbgemm amx_example gemm_amx
