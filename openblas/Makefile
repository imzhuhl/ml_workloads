OPENBLAS_INCLUDE=/hostfs/opt/OpenBLAS/include
OPENBLAS_LIB=/hostfs/opt/OpenBLAS/lib

COMPILE_OPTIONS=-I$(OPENBLAS_INCLUDE) -L$(OPENBLAS_LIB) -Wl,-rpath,$(OPENBLAS_LIB) -lopenblas

sgemm: sgemm.cpp utils.hpp
	g++ -O3 sgemm.cpp $(COMPILE_OPTIONS) -o sgemm

sbgemm: sbgemm.cpp
	g++ -O3 -march=armv8.6-a sbgemm.cpp $(COMPILE_OPTIONS) -o sbgemm

clean:
	rm sbgemm sgemm
