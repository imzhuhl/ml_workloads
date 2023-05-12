import time
import os
import torch
import argparse
from transformers import BertTokenizerFast, BertForMaskedLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch_size")
    parser.add_argument("--steps", type=int, default=10, help="num_steps")
    parser.add_argument("--threads", type=int, default=8, help="num_threads")
    parser.add_argument("--autocast", type=str, default=None)
    args = parser.parse_args()

    datatype = torch.float32
    if args.autocast == "bf16":
        datatype = torch.bfloat16

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    text = ['Paris is the [MASK] of France.' for _ in range(args.batch)]

    encoded_input = tokenizer(text, return_tensors='pt')
    model(**encoded_input)

    def perf_inference():
        t0 = time.time()
        for i in range(args.steps):
            encoded_input = tokenizer(text, return_tensors='pt')
            pred = model(**encoded_input)

            logits = pred.logits
            mask_token_index = (encoded_input.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            predicted_token_id = logits[mask_token_index].argmax(axis=-1)
            result = tokenizer.batch_decode(predicted_token_id)
        t1 = time.time()

        print(f"Throughput: {args.steps * args.batch / (t1 - t0)}")

    if args.autocast == "bf16":
        with torch.autocast(device_type='cpu', dtype=datatype):
            perf_inference()
    else:
        perf_inference()
