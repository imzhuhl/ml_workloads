#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "oneapi/dnnl/dnnl.hpp"
#include "utils.hpp"


void write_to_dnnl_memory(void *data_ptr, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (dst == nullptr) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; i++) {
      dst[i] = ((uint8_t *) data_ptr)[i];
    }
    return;
  }
  std::cerr << "not expected\n";
  throw std::runtime_error("incorrect engine type");
}


void run(int m, int n, int k) {
  // create engine
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);

  // create stream
  dnnl::stream engine_stream(engine);

  dnnl::memory::dims a_dims = {m, k};
  dnnl::memory::dims b_dims = {k, n};
  dnnl::memory::dims c_dims = {m, n};

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  std::vector<float> c_data(m * n);
  
  fill_array(a_data, InitValFlag::RandonValue);
  fill_array(b_data, InitValFlag::RandonValue);

  // Create memory descriptors and memory objects
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
  auto a_md = dnnl::memory::desc(a_dims, dt::f32, tag::any);
  auto b_md = dnnl::memory::desc(b_dims, dt::f32, tag::any);
  auto c_md = dnnl::memory::desc(c_dims, dt::f32, tag::any);

  auto a_in_md = dnnl::memory::desc(a_dims, dt::f32, tag::ab);
  auto b_in_md = dnnl::memory::desc(b_dims, dt::f32, tag::ab);
  auto a_in_mem = dnnl::memory(a_in_md, engine);
  auto b_in_mem = dnnl::memory(b_in_md, engine);

  // Write data to memory object's handles.
  write_to_dnnl_memory(a_data.data(), a_in_mem);
  write_to_dnnl_memory(b_data.data(), b_in_mem);

  // Create operation descriptor
  auto matmul_d = dnnl::matmul::desc(a_md, b_md, c_md);

  // Create primitive descriptor.
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

  // Repack and convert input data.
  auto a_mem = dnnl::memory(matmul_pd.src_desc(), engine);
  dnnl::reorder(a_in_mem, a_mem).execute(engine_stream, a_in_mem, a_mem);

  auto b_mem = dnnl::memory(matmul_pd.weights_desc(), engine);
  dnnl::reorder(b_in_mem, b_mem).execute(engine_stream, b_in_mem, b_mem);

  auto c_mem = dnnl::memory(matmul_pd.dst_desc(), engine);

  // Create the primitive.
  auto matmul_prim = dnnl::matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
  matmul_args.insert({DNNL_ARG_DST, c_mem});

  // 
  double num_gflop = 2.0 * m * n * k * 1.0e-09;
  double best_gflops = 0;

  // Warmup executions.
  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();

  for (int i = 0; i <= 5; i++) {
    auto start = std::chrono::steady_clock::now();
    matmul_prim.execute(engine_stream, matmul_args);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double cur_gflops = num_gflop / (elapsed.count() * 1.0e-3);
    best_gflops = cur_gflops > best_gflops ? cur_gflops : best_gflops;
  }

  printf("TARGET: %.2lf GFLOP/S\n", best_gflops);

}

void bad_args() {
    std::cerr << "Usage: sgemm m n k [-v]\n"
                 "       sgemm size [-v]\n"
                 "If a single <size> is specified, it is used for all three "
                 "dimensions (m/n/k).\n";
    throw std::invalid_argument("Incorrect input arguments.");
}


int main(int argc, char **argv) {
  printf("argc: %d\n", argc);

  int m, n, k;
  bool verbose = false;
  if (strcmp("-v", argv[argc-1]) == 0) {
    verbose = true;
    argc--;
  }
  if (argc == 2) {
    m = n = k = std::atoi(argv[1]);
  } else if (argc == 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  } else {
    bad_args();
  }

  printf("MatMul: m, n, k: %d, %d, %d\n", m, n, k);
  run(m, n, k);

  return 0;
}
