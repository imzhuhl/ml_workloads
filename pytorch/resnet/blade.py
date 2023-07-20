import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
import time
from utils import load_image
import torch_blade


def run_model(model, x):
    def run():
        model(x)

        s = time.time()
        for i in range(args.steps):
            pred = model(x)
        e = time.time()
        print(args.batch * args.steps / (e - s))
    
    if args.bf16:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            run()
    else:
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    parser.add_argument("--steps", type=int, default=10, help="steps")
    parser.add_argument("--bf16", action="store_true", help="use bf16")
    args = parser.parse_args()

    model = resnet50().eval()
    x = torch.randn((args.batch, 3, 224, 224))

    with torch.no_grad():
        blade_model = torch_blade.optimize(model, allow_tracing=True, model_inputs=(x,))

        run_model(blade_model, x)
    