import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
import time
from utils import load_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    args = parser.parse_args()

    model = resnet50()
    x = load_image(args.batch)

    print("JIT:")
    model.eval()

    traced_model = torch.jit.trace(model, x)
    traced_model = torch.jit.freeze(traced_model)
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    thp_lst = []
    
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
