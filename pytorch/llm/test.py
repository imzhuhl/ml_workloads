import time
import torch

# torch.random.manual_seed(42)
a = torch.randn((10000, 9))
# print(a)

t0 = time.time()
for _ in range(100):
# while True:
    # b = torch.tanh(a)
    b = torch.softmax(a, -1)
t1 = time.time()
print(t1 - t0)
# print(b)