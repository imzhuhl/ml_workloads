import torch
import intel_extension_for_pytorch as ipex
import argparse
from torchvision.models import resnet50, ResNet50_Weights
import time
from utils import load_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    args = parser.parse_args()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    x = load_image(args.batch)

    print("IPEX + JIT:")
    model.eval()

    model = ipex.optimize(model, level="O0")

    traced_model = torch.jit.trace(model, x)
    traced_model = torch.jit.freeze(traced_model)
    
    thp_lst = []
    with torch.no_grad():
        # warm up
        for i in range(3):
            traced_model(x)

        for i in range(10):
            s = time.time()
            pred = traced_model(x)
            e = time.time()
            thp_lst.append(args.batch / (e - s))
    
    thp_lst.sort()
    print(thp_lst[len(thp_lst)//2])
