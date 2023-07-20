import argparse
import torch
import torch._dynamo as dynamo
from torchvision.models import resnet50, ResNet50_Weights
import time
from utils import load_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch_size")
    parser.add_argument("--bf16", action="store_true", help="use bf16")
    args = parser.parse_args()

    data_type = torch.float32
    if args.bf16:
        data_type = torch.bfloat16

    steps = 10

    print(torch._dynamo.list_backends())

    model = resnet50()
    x = torch.randn((args.batch, 3, 224, 224)).to(data_type)

    @torch.no_grad()
    def run():
        model(x)

        s = time.time()
        for i in range(steps):
            pred = model(x)
        e = time.time()
        print(args.batch * steps / (e - s))


    model.eval()
    model = torch.compile(model)

    if args.bf16:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            run()
    else:
        run()
