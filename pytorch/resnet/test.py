import time
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import os

from utils import load_image


def tensor_to_channels_last_1d(t):
    assert t.dim() == 3

    if 1 == t.size(1):
        t = t.as_strided(t.size(), (t.size(1) * t.size(-1), 1, t.size(1)))
    else:
        t = t.view(t.size(0), t.size(1), 1, t.size(2))
        t = t.to(memory_format=torch.channels_last)
        t = t.view(t.size(0), t.size(1), t.size(3))
    return t

# Port from https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/blob/920c7163c81d6c5098ba79ed482d57b1ded8521d/intel_extension_for_pytorch/xpu/utils.py#L6 and
def to_channels_last_1d(t):
    if isinstance(t, torch.nn.Module):
        for m in t.modules():
            for param in m.parameters():
                if isinstance(m, (torch.nn.Conv1d)):
                    if 3 == param.data.dim():
                        param.data = tensor_to_channels_last_1d(param.data)
        return t

    if 3 == t.dim():
        t = tensor_to_channels_last_1d(t)
    return t

def _convert_convNd_weight_memory_format(module):
    # inspired from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/memory_format.py
    if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d):
        if isinstance(module, torch.nn.Conv1d):
            weight_data = to_channels_last_1d(module.weight.detach().clone())
            module.weight.data = weight_data.resize_(weight_data.size())
        elif isinstance(module, torch.nn.Conv2d):
            weight_data = module.weight.detach().clone().contiguous(memory_format=torch.channels_last)
            module.weight.data = weight_data.resize_(weight_data.size(), memory_format=torch.channels_last)
        elif isinstance(module, torch.nn.Conv3d):
            weight_data = module.weight.detach().clone().contiguous(memory_format=torch.channels_last_3d)
            module.weight.data = weight_data.resize_(weight_data.size(), memory_format=torch.channels_last_3d)

    for child in module.children():
        _convert_convNd_weight_memory_format(child)


def test(model, x):
    # import intel_extension_for_pytorch as ipex
    # from torch.profiler import profile, ProfilerActivity

    model = model.eval()
    # model = ipex.optimize(model, level="O1")
    # print(model)
    # _convert_convNd_weight_memory_format(model)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)
        # print(traced_model.graph)

        traced_model = torch.jit.freeze(traced_model)
        # traced_model = torch.jit.optimize_for_inference(traced_model)

        for i in range(3):
            pred = traced_model(x)
            # model(x)
        
        # print(traced_model.graph)

        # print("Profiling...")
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            # traced_model(x)
    #         # model(x)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))


class net_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class net_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=3, padding=2, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    args = parser.parse_args()
    print("Built with oneDNN:", torch.backends.mkldnn.is_available())

    x = load_image(args.batch)

    # model = net_1()
    model = net_2()
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    test(model, x)
    # perf(model, x)

    # perf_resnet.print_model()
    # perf_resnet.profiling()
