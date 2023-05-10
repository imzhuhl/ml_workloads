OPENBLAS_INCLUDE=/opt/OpenBLAS/include
OPENBLAS_LIB=/opt/OpenBLAS/lib

COMPILE_OPTIONS=-I$(OPENBLAS_INCLUDE) -L$(OPENBLAS_LIB) -Wl,-rpath,$(OPENBLAS_LIB) -lopenblas

sgemm: sgemm.cpp utils.hpp
	g++ -O2 sgemm.cpp $(COMPILE_OPTIONS) -o sgemm

sbgemm: sbgemm.cpp utils.hpp
	g++ -O2 sbgemm.cpp $(COMPILE_OPTIONS) -o sbgemm

test: compare_sgemm_sbgemm.cpp utils.hpp
	g++ -O2 compare_sgemm_sbgemm.cpp $(COMPILE_OPTIONS) -o compare_sgemm_sbgemm

clean:
	rm sbgemm sgemm compare_sgemm_sbgemm
