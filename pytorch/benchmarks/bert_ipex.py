import sys
import time
import torch
import intel_extension_for_pytorch as ipex
from transformers import BertTokenizerFast, BertModel
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
            model(**encoded_input)
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
    parser.add_argument("--dtype", choices=["fp32", "bf16", "amp"], default="fp32")
    parser.add_argument("--opt", choices=["normal", "dynamo", "jit"], default="normal")
    parser.add_argument("--short", action="store_true", help="use short seq")
    args = parser.parse_args()

    long_str = "This model is also a PyTorch torch.nn.Module subclass. \
                Use it as a regular PyTorch Module and refer to the PyTorch \
                documentation for all matter related to general usage and behavior."
    short_str = "Paris is the capital of [MASK]."

    if args.short:
        text = [short_str] * args.batch
    else:
        text = [long_str] * args.batch

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = dict(encoded_input)
    
    model.eval()
    
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

    with torch.no_grad():
        if args.dtype == "bf16":
            model = ipex.optimize(model, dtype=torch.bfloat16)
        else:
            model = ipex.optimize(model)

        if args.opt == "dynamo":
            model = torch.compile(model, backend="ipex")
        elif args.opt == "jit":
            model = torch.jit.trace(model, example_kwarg_inputs=encoded_input, strict=False)
            model = torch.jit.freeze(model)
        
        if args.dtype == "amp":
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                run_inference(model, encoded_input)
        else:
            run_inference(model, encoded_input)
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     run(model, encoded_input)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))

