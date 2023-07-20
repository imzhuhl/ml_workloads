import torch
import time

# [batch, length, 768] -> [batch * length, 768]

dtype = torch.float32

matrix_x = torch.randn((12544, 768)).to(dtype)

wk = torch.randn((768, 768)).to(dtype)
wq = torch.randn((768, 768)).to(dtype)
wv = torch.randn((768, 768)).to(dtype)

wall = torch.randn((768, 768*3)).to(dtype)

steps = 100

for _ in range(steps):
    matrix_x @ wk
    matrix_x @ wq
    matrix_x @ wv
    matrix_x @ wall


t0 = time.time()
for _ in range(steps):
    matrix_x @ wall
t1 = time.time()
print(t1 - t0)

print("=====")

t0 = time.time()
for _ in range(steps):
    matrix_x @ wk
    matrix_x @ wq
    matrix_x @ wv
t1 = time.time()
print(t1 - t0)





