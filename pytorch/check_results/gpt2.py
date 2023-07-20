import sys
import time
import torch
import argparse
from transformers import GPT2TokenizerFast, GPT2Model


def run_and_save_result(name):
    long_str = "This model is also a PyTorch torch.nn.Module subclass. \
                    Use it as a regular PyTorch Module and refer to the PyTorch \
                    documentation for all matter related to general usage and behavior"
    short_str = "Paris is the capital of"

    text = [short_str]

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', local_files_only=True)
    model = GPT2Model.from_pretrained('gpt2', local_files_only=True)

    model.eval()

    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = dict(encoded_input)

    # model = torch.compile(model)
    with torch.no_grad():
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            pred = model(**encoded_input)
    
    pred = dict(pred)
    pred = pred['last_hidden_state']
    torch.save(pred, f"{name}.pt")


def compare(name1, name2):
    rst1 = torch.load(f"{name1}.pt")
    rst2 = torch.load(f"{name2}.pt")

    assert isinstance(rst1, torch.Tensor)
    rst = torch.where(rst1.abs() - rst2.abs() > 1e-3, 1, 0).sum()
    print(rst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str)
    parser.add_argument("--compare", nargs='+', type=str)
    args = parser.parse_args()

    if args.save:
        run_and_save_result(args.save)

    if args.compare:
        tmp = [item.strip() for item in ','.join(args.compare).split(',')]
        compare(tmp[0], tmp[1])

