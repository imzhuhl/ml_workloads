import time
import os
import torch
import argparse
from transformers import pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch_size")
    parser.add_argument("--steps", type=int, default=10, help="num_steps")
    parser.add_argument("--threads", type=int, default=8, help="num_threads")
    args = parser.parse_args()

    pipe = pipeline('fill-mask', model='bert-base-uncased')

    text = ['Paris is the [MASK] of France.' for _ in range(args.batch)]

    pipe(text)

    t0 = time.time()
    for i in range(args.steps):
        pipe(text)
    t1 = time.time()
    print(f"Throughput: {args.steps * args.batch / (t1 - t0)}")

