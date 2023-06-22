import torch
import torch._dynamo as torchdynamo
from torch._functorch.aot_autograd import aot_module_simplified
from typing import List


def toy_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("my_compiler() called with FX graph:")
        gm.print_readable()
        return gm.forward  # return a python callable

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=my_compiler
    )


def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def run():
    for _ in range(1):
        toy_example(torch.randn(10), torch.randn(10))


def foo_fn():
    print("foo_fn() called")

def foo1(fn):
    def foo2():
        print("foo2() called")

    foo2._bbbb = fn
    print("foo1() called")
    return foo2


def simple_test(a, b):
    c = a + b
    return c



if __name__ == "__main__":
    # foo1(foo_fn)
    # toy_example(torch.randn(10), torch.randn(10))
    simple_test = torch.compile(simple_test, backend=toy_backend)
    simple_test(torch.randn(5, 5), torch.randn(5, 5))

