import sys
import time
import torch
from PIL import Image
from transformers import GPT2TokenizerFast, GPT2Model, BertTokenizerFast, BertModel
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DRYRUN = 'dryrun' in sys.argv

@torch.no_grad()
def run_model(name, tokenizer, model, input, steps):
    def run():
        encoded_input = tokenizer(input, return_tensors='pt')
        for _ in range(3): model(**encoded_input)  # warmup
        
        t0 = time.time()
        for _ in range(steps):
            encoded_input = tokenizer(input, return_tensors='pt')
            model(**encoded_input)
        return steps * len(input) / (time.time() - t0)

    if DRYRUN:
        return

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        return run()


text = "Use it as a regular PyTorch Module and refer to the PyTorch \
        documentation for all matter related to general usage and behavior"

data = []
for _ in range(5):
    th = run_model('bert-base-uncased',
            BertTokenizerFast.from_pretrained(
                'bert-base-uncased', local_files_only=not DRYRUN),
            BertModel.from_pretrained(
                'bert-base-uncased', local_files_only=not DRYRUN),
            ['Paris is the capital of [MASK].']*256, 10)
    data.append(th)
print(f"bert avg: {sum(data) / len(data)}")
print(f"bert max: {max(data)}")


data = []
for _ in range(5):
    th = run_model('gpt2',
            GPT2TokenizerFast.from_pretrained(
                'gpt2', local_files_only=not DRYRUN),
            GPT2Model.from_pretrained('gpt2', local_files_only=not DRYRUN),
            ['Once upon a time,']*256, 10)
    data.append(th)
print(f"gpt2 avg: {sum(data) / len(data)}")
print(f"gpt2 max: {max(data)}")
