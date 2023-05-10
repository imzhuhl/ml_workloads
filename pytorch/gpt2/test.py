import time
import torch
from transformers import pipeline


def run_model(name, num, inference):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inference()  # warmup
        t0 = time.time()
        for i in range(num):
            inference()
        print(name + ' ' + str(num / (time.time() - t0)))


unmasker = pipeline('fill-mask', model='bert-base-uncased')
run_model('bert-base-uncased', 1000,
          lambda: unmasker('Paris is the [MASK] of France.'))

generator = pipeline('text-generation', model='gpt2')
run_model('gpt2', 10, lambda: generator(
    'Once upon a time,', max_length=30, num_return_sequences=5))
