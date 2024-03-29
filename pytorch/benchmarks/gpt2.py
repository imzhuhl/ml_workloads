import sys
import time
import torch
from PIL import Image
from transformers import GPT2TokenizerFast, GPT2Model
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help='batch size')
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "amp"], default="fp32")
    parser.add_argument("--opt", choices=["normal", "dynamo", "jit"], default="normal")
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

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    model.eval()

    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = dict(encoded_input)

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)

    def run_inference(model, encoded_input):
        throughput = []
        for i in range(args.steps):
            t0 = time.time()
            pred = model(**encoded_input)
            t1 = time.time()
            print(f"steps: {i} | time(ms): {(t1 - t0) * 1000:.2f} | throughput: {args.batch / (t1 - t0):.3f}")
            if i > 1: throughput.append(args.batch / (t1 - t0))
        avg_throughput = sum(throughput) / len(throughput)
        print(f"avg throughput: {avg_throughput:.3f}")

    def profile(model, encoded_input):
        from torch.profiler import profile, ProfilerActivity, schedule
        my_schedule = schedule(skip_first=5, wait=5, warmup=5, active=1, repeat=5)
        with profile(activities=[ProfilerActivity.CPU], schedule=my_schedule) as prof:
            for i in range(20):
                pred = model(**encoded_input)
                prof.step()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20, top_level_events_only=False))

    with torch.no_grad():
        if args.opt == "dynamo":
            model = torch.compile(model)
        elif args.opt == "jit":
            model = torch.jit.trace(model, example_kwarg_inputs=encoded_input, strict=False)
            model = torch.jit.freeze(model)
            model = torch.jit.optimize_for_inference(model)

        if args.dtype == "amp":
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                run_inference(model, encoded_input)
                # profile(model, encoded_input)
        else:
            run_inference(model, encoded_input)
            # profile(model, encoded_input)
        

# from torch._inductor import config
# config.cpp.enable_kernel_profile = True
main()
