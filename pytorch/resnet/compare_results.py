import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import numpy as np
from utils import load_image


def compare1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch_size")
    args = parser.parse_args()

    model = resnet50(ResNet50_Weights.DEFAULT)
    x = load_image(args.batch)

    model.eval()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        rst1 = model(x)

    # with torch.no_grad():
    #     rst2 = model(x)

    compiled_model = torch.compile(model)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        rst2 = compiled_model(x)
    
    diff = torch.abs(rst2 - rst1)

    print(torch.median(diff))
    item = torch.sort(diff, descending=True)
    ids = item.indices
    vals = item.values
    
    vals = torch.flatten(vals)
    ids = torch.flatten(ids)

    for i in range(5):
        print(vals[i], rst1[0, ids[i]], rst2[0, ids[i]])


def compare2():
    preprocess = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
    model.eval()

    image = Image.open('./img/ILSVRC2012_val_00000002.JPEG')
    encoded_input = preprocess(image, return_tensors='pt')

    # with torch.no_grad():
    #     rst1 = model(**encoded_input).logits

    with torch.autocast("cpu", dtype=torch.bfloat16):
        rst1 = model(**encoded_input).logits

    compiled_model = torch.compile(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rst2 = compiled_model(**encoded_input).logits

    diff = torch.abs(rst2 - rst1)
    print(torch.median(diff))
    item = torch.sort(diff, descending=True)
    ids = item.indices
    vals = item.values
    vals = torch.flatten(vals)
    ids = torch.flatten(ids)
    for i in range(5):
        print(vals[i], rst1[0, ids[i]], rst2[0, ids[i]])


compare2()
