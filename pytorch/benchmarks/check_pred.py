import sys
import time
import torch
from PIL import Image
from transformers import GPT2TokenizerFast, GPT2Model
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def run_v1(name, tokenizer, model, input):
    model.eval()
    model = torch.compile(model)

    def run():
        encoded_input = tokenizer(input, return_tensors='pt')
        for i in range(args.steps):
            pred = model(**encoded_input)
        return pred

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        return run()


@torch.no_grad()
def run_v2(name, tokenizer, model, input):
    import intel_extension_for_pytorch as ipex

    model.eval()
    model = ipex.optimize(model)
    model = torch.compile(model, backend="ipex")

    def run():
        encoded_input = tokenizer(input, return_tensors='pt')
        for i in range(args.steps):
            pred = model(**encoded_input)
        return pred

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        return run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256, help='batch size')
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    long_str = "This model is also a PyTorch torch.nn.Module subclass. \
                Use it as a regular PyTorch Module and refer to the PyTorch \
                documentation for all matter related to general usage and behavior"
    text = [long_str] * args.batch

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    pred_v1 = run_v1('gpt2', tokenizer, model, text)
    pred_v2 = run_v2('gpt2', tokenizer, model, text)

    print(pred_v1)


