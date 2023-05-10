#include <chrono>
#include <iostream>
#include <vector>
#include "utils.hpp"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Scheduler.h"

using namespace arm_compute;

void fill_acl_tensor(Tensor &tensor, float *data, size_t data_size) {
    assert(tensor.info()->tensor_shape().total_size() == data_size);
    // map(tensor, true);

    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    int i = 0;
    Iterator it_tensor(&tensor, window);
    execute_window_loop(
        window,
        [&](const Coordinates &) {
            *reinterpret_cast<float *>(it_tensor.ptr()) = data[i++];
        },
        it_tensor);

    // unmap(tensor);
}

void read_acl_tensor(Tensor &tensor, float *data, size_t data_size) {
    assert(tensor.info()->tensor_shape().total_size() == data_size);
    // map(tensor, true);

    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    int i = 0;
    Iterator it_tensor(&tensor, window);
    execute_window_loop(
        window,
        [&](const Coordinates &) {
            data[i++] = *reinterpret_cast<float *>(it_tensor.ptr());
        },
        it_tensor);

    // unmap(tensor);
}

int run_sbgemm(int *mat_size, bool verbose) {
    int M, N, K;
    if (mat_size[0] != 0 && mat_size[1] == 0 && mat_size[2] == 0) {
        M = mat_size[0];
        N = M;
        K = M;
    } else if (mat_size[0] != 0 && mat_size[1] != 0 && mat_size[2] != 0) {
        M = mat_size[0];
        N = mat_size[1];
        K = mat_size[2];
    }

    double gflops = 2.0 * M * N * K * 1.0e-9;

    float alpha = 1;
    float beta = 0;

    float *FA = new float[M * K];
    float *FB = new float[K * N];
    float *C = new float[M * N];
    float *myc = new float[M * N];

    fill_vector(FA, M * K, InitVecFlag::RandonValue);
    fill_vector(FB, K * N, InitVecFlag::RandonValue);
    fill_vector(C, M * N, InitVecFlag::Zero);

    Tensor src0{}, src1{}, dst{};
    NEGEMM sbgemm{};
    NEGEMM sgemm{};

    src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
    src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
    dst.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

    GEMMInfo sgemm_info{}, sbgemm_info{};
    sgemm_info.set_fast_math(false);
    sbgemm_info.set_fast_math(true);

    sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta, sgemm_info);
    sbgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta, sbgemm_info);

    src0.allocator()->allocate();
    src1.allocator()->allocate();
    dst.allocator()->allocate();

    std::vector<double> rst;
    for (int rep = 0; rep < 7; rep++) {
        fill_acl_tensor(src0, FA, M * K);
        fill_acl_tensor(src1, FB, K * N);
        fill_acl_tensor(dst, myc, M * N);
        auto st = std::chrono::steady_clock::now();
        sbgemm.run();  // matmul
        auto et = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = et - st;
        double time_s = elapsed.count() * 1.0e-3;
        printf("%.3lf GFLOP/S, %.2lf ms\n", gflops / time_s, elapsed.count());
        rst.push_back(gflops / time_s);
    }
    // save result to myc;
    read_acl_tensor(dst, myc, M * N);
    
    // check result
    fill_acl_tensor(src0, FA, M * K);
    fill_acl_tensor(src1, FB, K * N);
    fill_acl_tensor(dst, C, M * N);
    sgemm.run();
    read_acl_tensor(dst, C, M * N);
    compare_matrix(C, myc, M, N);

    // performance
    std::sort(rst.begin(), rst.end(), std::greater<double>());
    double best = rst[0];
    printf("Best: %.3lf GFLOP/S\n", best);

    if (verbose) {
        printf("Matrix A:\n");
        display_matrix(FA, M * K, K, MAT_ORDER::RowMajor);
        printf("Matrix B:\n");
        display_matrix(FB, K * N, N, MAT_ORDER::RowMajor);
        printf("Matrix C:\n");
        display_matrix(C, M * N, N, MAT_ORDER::RowMajor);
        printf("Matrix myc:\n");
        display_matrix(myc, M * N, N, MAT_ORDER::RowMajor);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    // Scheduler::get().set_num_threads(1);
    if (argc <= 1 || argc > 5) {
        printf("Please run:\n1./sgemm mat_size [-v]\n./sgemm m n k [-v]");
        return 0;
    }
    bool verbose = false;
    int mat_size[3] = {0};
    int idx = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp("-v", argv[i]) == 0) {
            verbose = true;
        } else {
            mat_size[idx++] = atoi(argv[i]);
        }
    }
    
    run_sbgemm(mat_size, verbose);
    return 0;
}