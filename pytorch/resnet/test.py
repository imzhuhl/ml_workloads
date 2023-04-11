import time
import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import argparse
import os
import copy

class PerfResnet:
    def __init__(self, model, input_x):
        self._model = model
        self._input = input_x
    
    def normal_inference(self):
        print("Normal:")
        self._model.eval()

        thp_lst = []
        with torch.no_grad():
            # warm up
            for _ in range(3):
                self._model(self._input)
            
            for i in range(10):
                s = time.time()
                pred = self._model(self._input)
                e = time.time()
                thp_lst.append(args.batch / (e - s))

        thp_lst.sort()
        print(thp_lst[len(thp_lst)//2])

    def jit_inference(self):
        print("JIT:")
        self._model.eval()
        model = copy.deepcopy(self._model)
        
        traced_model = torch.jit.trace(model, self._input)
        traced_model = torch.jit.freeze(traced_model)
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        thp_lst = []
        
        # warm up
        for i in range(3):
            traced_model(self._input)

        for i in range(10):
            s = time.time()
            pred = traced_model(self._input)
            e = time.time()
            thp_lst.append(args.batch / (e - s))
        
        thp_lst.sort()
        print(thp_lst[len(thp_lst)//2])

    def ipex_inference(self):
        print("IPEX:")
        model = copy.deepcopy(self._model)

        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, level="O1")

        thp_lst = []
        with torch.no_grad():
            # warm up
            for _ in range(3):
                model(self._input)
            
            for i in range(10):
                s = time.time()
                pred = model(self._input)
                e = time.time()
                thp_lst.append(args.batch / (e - s))

        thp_lst.sort()
        print(thp_lst[len(thp_lst)//2])

    def ipex_jit_inference(self):
        print("IPEX + JIT:")
        self._model.eval()
        model = copy.deepcopy(self._model)

        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, level="O1")

        with torch.no_grad():
            traced_model = torch.jit.trace(model, self._input)
            traced_model = torch.jit.freeze(traced_model)
            # traced_model = torch.jit.optimize_for_inference(traced_model)
            
            thp_lst = []

            # warm up
            for i in range(3):
                traced_model(self._input)

            for i in range(10):
                s = time.time()
                pred = traced_model(self._input)
                e = time.time()
                thp_lst.append(args.batch / (e - s))
            
        thp_lst.sort()
        print(thp_lst[len(thp_lst)//2])

    def print_model(self):
        self._model.eval()

        # import intel_extension_for_pytorch as ipex
        # model = ipex.optimize(self._model)
        # print(model)

        # traced_model = torch.jit.trace(model, self._input)
        # traced_model = torch.jit.freeze(traced_model)

        # with torch.no_grad():
        #     for i in range(3):
        #         traced_model(self._input)

        # print(traced_model.graph_for(self._input))

        pass
    
    def profiling(self):
        import intel_extension_for_pytorch as ipex
        from torch.profiler import profile, ProfilerActivity

        model = self._model.eval()
        # model = ipex.optimize(model, level="O0")
        # print(model)
        # _convert_convNd_weight_memory_format(model)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, self._input)
            # traced_model = torch.jit.freeze(traced_model)
            # traced_model = torch.jit.optimize_for_inference(traced_model)

            for i in range(3):
                pred = traced_model(self._input)
                # model(self._input)

            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                pred = traced_model(self._input)
                # model(self._input)
        
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15, top_level_events_only=False))



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


def load_image():
    img = Image.open("data/ILSVRC2012_val_00000002.JPEG").convert("RGB")
    resized_img = img.resize((224, 224))
    img_data = np.asarray(resized_img).astype("float32")
    img_data = np.transpose(img_data, (2, 0, 1))  # CHW

    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW
    img_data = np.expand_dims(norm_img_data, axis=0).repeat(args.batch, axis=0)
    x = torch.from_numpy(img_data).to(torch.float32)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    args = parser.parse_args()
    print("Built with oneDNN:", torch.backends.mkldnn.is_available())

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    x = load_image()
    perf_resnet = PerfResnet(model, x)

    perf_resnet.normal_inference()
    perf_resnet.ipex_inference()
    perf_resnet.jit_inference()
    perf_resnet.ipex_jit_inference()
    # perf_resnet.print_model()
    # perf_resnet.profiling()


