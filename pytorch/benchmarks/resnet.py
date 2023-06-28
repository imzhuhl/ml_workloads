import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
import time

@torch.no_grad()
def run_model(name, model, x):
    model.eval()

    if args.dynamo:
        model = torch.compile(model)

    def run():
        throughput = []
        for i in range(args.steps):
            t0 = time.time()
            pred = model(x)
            t1 = time.time()
            print(f"steps: {i} | time(ms): {(t1 - t0) * 1000:.2f} | throughput: {args.batch / (t1 - t0):.3f}")
            if i > 1: throughput.append(args.batch / (t1 - t0))
        avg_throughput = sum(throughput) / len(throughput)
        print(f"avg throughput: {avg_throughput:.3f}")
    
    if args.cast:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            run()
    else:
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help='batch size')
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cast", action="store_true", help="autocast")
    parser.add_argument("--dynamo", action="store_true", help="use torchdynamo")
    args = parser.parse_args()

    model = resnet50()
    x = torch.randn((args.batch, 3, 224, 224))
    
    run_model('resnet50', model, x)
    