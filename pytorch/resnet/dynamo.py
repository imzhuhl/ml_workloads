import argparse
import torch
import torch._dynamo as dynamo
from torchvision.models import resnet50, ResNet50_Weights
import time
from utils import load_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    args = parser.parse_args()

    print(torch._dynamo.list_backends())

    model = resnet50()
    x = torch.randn((args.batch, 3, 224, 224))


    model.eval()
    model = torch.compile(model)

    thp_lst = []
    with torch.no_grad():
        # warm up
        for _ in range(3):
            model(x)
        
        for i in range(10):
            s = time.time()
            pred = model(x)
            e = time.time()
            print(f"Time (ms): {(e - s) * 1000}")
            thp_lst.append(args.batch / (e - s))

    thp_lst.sort()
    print(thp_lst[len(thp_lst)//2])
