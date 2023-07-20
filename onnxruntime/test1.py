import torch
import torch.nn as nn
import onnx
import onnxruntime
import argparse


class MyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


def export_onnx_model():
    model = MyNet()
    x = torch.randn(2, 64)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "mynet.onnx",
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


def run_onnx_model():
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = True
    session = onnxruntime.InferenceSession(
        "./mynet.onnx", sess_options, providers=["CPUExecutionProvider"]
    )

    x = torch.randn(3, 64)
    inputs = {session.get_inputs()[0].name: x.numpy()}
    outs = session.run(None, inputs)
    print(outs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="export onnx model")
    args = p.parse_args()

    if args.export:
        export_onnx_model()
    else:
        run_onnx_model()
