import sys
import time
import torch
import intel_extension_for_pytorch as ipex
from PIL import Image
from transformers import GPT2TokenizerFast, GPT2Model
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def run_model(name, tokenizer, model, input):
    model.eval()
    model = ipex.optimize(model)
    model = torch.compile(model, backend="ipex")

    def run():
        encoded_input = tokenizer(input, return_tensors='pt')
        throughput = []
        for i in range(args.steps):
            t0 = time.time()
            pred = model(**encoded_input)
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
    parser.add_argument("--batch", type=int, default=256, help='batch size')
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cast", action="store_true", help="autocast")
    parser.add_argument("--short", action="store_true", help="use short seq")
    args = parser.parse_args()

    long_str = "This model is also a PyTorch torch.nn.Module subclass. \
                Use it as a regular PyTorch Module and refer to the PyTorch \
                documentation for all matter related to general usage and behavior"
    short_str = "Paris is the capital of"

    if args.short:
        text = [short_str] * args.batch
    else:
        text = [long_str] * args.batch

    run_model('gpt2',
              GPT2TokenizerFast.from_pretrained('gpt2'),
              GPT2Model.from_pretrained('gpt2'),
              text)
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))

