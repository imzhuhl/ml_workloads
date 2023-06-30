import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help='batch size')
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "amp"], default="fp32")
    parser.add_argument("--opt", choices=["normal", "dynamo", "jit"], default="normal")
    args = parser.parse_args()

    model = resnet50().eval()
    x = torch.randn((args.batch, 3, 224, 224))
    
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)

    def run_inference(model, x):
        throughput = []
        for i in range(args.steps):
            t0 = time.time()
            pred = model(x)
            t1 = time.time()
            print(f"steps: {i} | time(ms): {(t1 - t0) * 1000:.2f} | throughput: {args.batch / (t1 - t0):.3f}")
            if i > 1: throughput.append(args.batch / (t1 - t0))
        avg_throughput = sum(throughput) / len(throughput)
        print(f"avg throughput: {avg_throughput:.3f}")

    with torch.no_grad():
        if args.opt == "dynamo":
            model = torch.compile(model)
        elif args.opt == "jit":
            model = torch.jit.trace(model, x)
            model = torch.jit.freeze(model)
            model = torch.jit.optimize_for_inference(model)

        if args.dtype == "amp":
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                run_inference(model, x)
        else:
            run_inference(model, x)


main()