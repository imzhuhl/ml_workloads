import time
import os
import torch
import argparse
from transformers import BertTokenizerFast, BertForMaskedLM


class PerfHandler:
    def __init__(self, args, tokenizer, model, text) -> None:
        self.batch_size = args.batch
        self.steps = args.steps
        self.threads = args.threads
        self.use_profiling = args.profiling
        self.autocast = args.autocast
        self.tokenizer = tokenizer
        self.model = model
        self.text = text

    def perf_inference(self):
        t0 = time.time()
        for i in range(self.steps):
            encoded_input = self.tokenizer(self.text, return_tensors='pt')
            pred = self.model(**encoded_input)

            # logits = pred.logits
            # mask_token_index = (encoded_input.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            # predicted_token_id = logits[mask_token_index].argmax(axis=-1)
            # result = self.tokenizer.batch_decode(predicted_token_id)
        t1 = time.time()

        print(f"Throughput: {self.steps * self.batch_size / (t1 - t0)}")


    def profiling(self):
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            self.perf_inference()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))


    def start_inference(self):
        encoded_input = self.tokenizer(self.text, return_tensors='pt')
        self.model(**encoded_input)

        if self.autocast == "bf16":
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                if self.use_profiling:
                    self.profiling()
                else:
                    self.perf_inference()
        else:
            if self.use_profiling:
                self.profiling()
            else:
                self.perf_inference()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch_size")
    parser.add_argument("--steps", type=int, default=5, help="num_steps")
    parser.add_argument("--threads", type=int, default=8, help="num_threads")
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--autocast", type=str, default=None)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    text = ['Paris is the [MASK] of France.' for _ in range(args.batch)]

    PerfHandler(args, tokenizer, model, text).start_inference()


if __name__ == '__main__':
    main()
