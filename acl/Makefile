CXX = g++
COMPILE_FLAG += -std=c++14

ifndef DEBUG
COMPILE_FLAG += -O2
else
COMPILE_FLAG += -g -DDEBUG
endif

ACL_REPO=/hostfs/lab/ai/ComputeLibrary
ACL_INCLUDE=/hostfs/lab/ai/ComputeLibrary/include
ACL_COMPUTE_LIB=/hostfs/lab/ai/ComputeLibrary/build

SGEMM_OPTIONS=-I$(ACL_REPO) -I$(ACL_INCLUDE) -L$(ACL_COMPUTE_LIB) -Wl,-rpath,$(ACL_COMPUTE_LIB) -larm_compute -larm_compute_core -larm_compute_graph

sgemm: utils.hpp sgemm.cpp
	$(CXX) $(COMPILE_FLAG) sgemm.cpp $(SGEMM_OPTIONS) -o sgemm

sbgemm: sbgemm.cpp
	$(CXX) $(COMPILE_FLAG) sbgemm.cpp $(SGEMM_OPTIONS) -o sbgemm

clean:
	rm sgemm sbgemm
