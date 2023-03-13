import numpy as np
import time


def bench_gemm(m, n, k):
    print(f"M, N, K: {m}, {n}, {k}")
    results = []
    gflops = 2.0 * m * n * k * 1.0e-09

    a = np.arange(float(m * k), dtype='float32').reshape(m, k)
    b = np.arange(float(k * n), dtype='float32').reshape(k, n)
    for i in range(50):
        t0 = time.time()
        np.matmul(a, b)
        t1 = time.time()
        rst = gflops / (t1 - t0)
        results.append(rst)

    results.sort()

    # calculate avg
    print(f"gemm{m} {results[len(results) // 2]} GFLOPS")


if __name__ == "__main__":
    bench_gemm(100, 100, 100)
    bench_gemm(200, 200, 200)
    bench_gemm(400, 400, 400)
    bench_gemm(800, 800, 800)
    bench_gemm(1600, 1600, 1600)
    bench_gemm(12544, 64, 147)
    bench_gemm(3136, 64, 147)
    bench_gemm(784, 128, 147)
    bench_gemm(784, 64, 147)


