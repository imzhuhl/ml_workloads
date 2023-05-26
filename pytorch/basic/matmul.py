import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bf16", type=bool, default=False)
args = parser.parse_args()

m, n, k = 2000, 2000, 2000
num_gflop = 2 * m * n * k

a = torch.randn((m, k))
b = torch.randn((k, n))

if args.bf16:
    a.to(torch.bfloat16)
    b.to(torch.bfloat16)

best_gflops = 0
for i in range(1000):
    t0 = time.time()
    c = torch.matmul(a, b)
    t1 = time.time()
    cur_gflops = num_gflop / (t1 - t0) / 1e9
    if best_gflops < cur_gflops:
        best_gflops = cur_gflops

print(f"GFLOPS: {best_gflops:.2f}")

