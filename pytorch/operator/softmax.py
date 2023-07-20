import torch
import torch.nn as nn
import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = nn.functional.softmax(x, dim=-1)
        return x


def perf():
    # self-attention softmax performance
    batch_size = 256
    num_attention_heads = 12
    seq_length = 7
    steps = 1000

    x = torch.randn((batch_size, num_attention_heads, seq_length, seq_length))

    model = Net()

    # model = torch.compile(model)

    y = model(x)

    t0 = time.time()
    for _ in range(steps):
        model(x)
    origin_time = time.time() - t0

    if not hasattr(torch, 'set_extra_optimization'):
        print("no set_extra_optimization")
        
    torch.set_extra_optimization(True)
    t0 = time.time()
    for _ in range(steps):
        model(x)
    opt_time = time.time() - t0

    print(origin_time, opt_time)
    scale = origin_time / opt_time
    print(scale)


def check():
    x = torch.randn((3, 7))
    y1 = nn.functional.softmax(x, dim=-1)
    # print(y1)

    if hasattr(torch, 'set_extra_optimization'):
        torch.set_extra_optimization(True)
        y2 = nn.functional.softmax(x, dim=-1)
        # print(y2)
        diff = torch.abs(y2 - y1).flatten()
        # print(diff)
        # sorted_diff, sorted_indices = torch.sort(diff, descending=True)
        print(torch.sum(diff > 1e-6))


# check()
perf()
