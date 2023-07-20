import time
import torch
import sys
import onnxruntime
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType
from resnet50_data_reader import ResNet50DataReader


def export_model():
    batch = 32
    model = resnet50().eval()
    x = torch.randn(batch, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "resnet50.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        # dr = ResNet50DataReader("./test_images", "resnet50.onnx")
        # quantize_static("resnet50.onnx", "resnet50_quant.onnx", dr)


def run_model(name, model_path, batch):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    x = torch.randn(batch, 3, 224, 224)

    ort_session = onnxruntime.InferenceSession(model_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    # warm up
    for _ in range(5):
        ort_outs = ort_session.run(None, ort_inputs)

    throughput = []
    for i in range(100):
        t0 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t1 = time.time()
        print(f"steps: {i} | time(ms): {(t1 - t0) * 1000:.2f} | throughput: {batch / (t1 - t0):.3f}")
        if i > 1: throughput.append(batch / (t1 - t0))
    avg_throughput = sum(throughput) / len(throughput)
    print(f"TARGET: {name} {avg_throughput:.3f}")


# export_model()
# run_model("resnet50_fp32_single", "resnet50.onnx", 1)
run_model("resnet50_fp32_batch", "resnet50.onnx", 32)
# run_model("resnet50_int8_single", "resnet50_quant.onnx", 1)
# run_model("resnet50_int8_batch", "resnet50_quant.onnx", 32)
