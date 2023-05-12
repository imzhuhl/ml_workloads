import time
import os
import torch
import argparse
from transformers import BertTokenizerFast, BertForMaskedLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256, help="batch_size")
    parser.add_argument("--steps", type=int, default=10, help="num_steps")
    parser.add_argument("--threads", type=int, default=8, help="num_threads")
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--autocast", type=str, default=None)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    text = ['Paris is the [MASK] of France.' for _ in range(args.batch)]

    encoded_input = tokenizer(text, return_tensors='pt')

    print(model.bert)

    # for _ in range(10):
    #     model(**encoded_input)

if __name__ == "__main__":
    main()
