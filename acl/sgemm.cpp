#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils.hpp"
#include "arm_compute/runtime/Scheduler.h"

using namespace arm_compute;

/** Maps a tensor if needed
 *
 * @param[in] tensor   Tensor to be mapped
 * @param[in] blocking Specified if map is blocking or not
 */
template <typename T>
inline void map(T &tensor, bool blocking) {
    ARM_COMPUTE_UNUSED(tensor);
    ARM_COMPUTE_UNUSED(blocking);
}

/** Unmaps a tensor if needed
 *
 * @param tensor  Tensor to be unmapped
 */
template <typename T>
inline void unmap(T &tensor) {
    ARM_COMPUTE_UNUSED(tensor);
}

void fill_acl_tensor(Tensor &tensor, float *data, size_t data_size) {
    assert(tensor.info()->tensor_shape().total_size() == data_size);
    map(tensor, true);

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

    unmap(tensor);
}

void read_acl_tensor(Tensor &tensor, float *data, size_t data_size) {
    assert(tensor.info()->tensor_shape().total_size() == data_size);
    map(tensor, true);

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

    unmap(tensor);
}

// #ifdef DEBUG
// constexpr size_t M = 4;
// constexpr size_t K = 4;
// constexpr size_t N = 4;
// #else
// constexpr size_t M = 1024;
// constexpr size_t K = 1040;
// constexpr size_t N = 1056;
// #endif

int run_sgemm(int *mat_size, bool verbose) {
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

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];
    float *myc = new float[M * N];

    fill_vector(A, M * K, InitVecFlag::RandonValue);
    fill_vector(B, K * N, InitVecFlag::RandonValue);
    fill_vector(C, M * N, InitVecFlag::Zero);

// #ifdef DEBUG
//     printf("Matrix A:\n");
//     display_matrix(A, M * K, K, MAT_ORDER::RowMajor);
//     printf("Matrix B:\n");
//     display_matrix(B, K * N, N, MAT_ORDER::RowMajor);
// #endif

    Tensor src0{}, src1{}, dst{};
    NEGEMM sgemm{};

    src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
    src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
    dst.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

    sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

    src0.allocator()->allocate();
    src1.allocator()->allocate();
    dst.allocator()->allocate();

    fill_acl_tensor(src0, A, M * K);
    fill_acl_tensor(src1, B, K * N);

    std::vector<double> rst;
    for (int rep = 0; rep < 10; rep++) {
        fill_acl_tensor(dst, C, M * N);
        auto st = std::chrono::steady_clock::now();
        sgemm.run();  // matmul
        auto et = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = et - st;
        double time_s = elapsed.count() * 1.0e-3;
        printf("%.3lf GFLOP/S, %.2lf ms\n", gflops / time_s, elapsed.count());
        rst.push_back(gflops / time_s);
    }

    read_acl_tensor(dst, C, M * N);

    // sort rst, make the largest to first
    std::sort(rst.begin(), rst.end(), std::greater<double>());
    double best = rst[0];
    printf("Best: %.3lf GFLOP/S\n", best);


    delete[] A;
    delete[] B;
    delete[] C;
    delete[] myc;

    return 0;
}

int main(int argc, char *argv[]) {
    // Scheduler::get().set_num_threads(8);
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

    run_sgemm(mat_size, verbose);
    return 0;
}