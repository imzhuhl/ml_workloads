import sys
import time
import torch
from PIL import Image
from transformers import BertTokenizerFast, BertModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cpu'
batch = 256


def run_model(name, tokenizer, model, input, steps):
    model.eval()
    model = torch.compile(model)

    @torch.no_grad()
    def run():
        encoded_input = tokenizer(input, return_tensors='pt')
        throughput = []
        for i in range(steps):
            t0 = time.time()
            model(**encoded_input)
            t1 = time.time()
            print(f"steps: {i} | time(ms): {(t1 - t0) * 1000:.2f} | throughput: {batch / (t1 - t0):.3f}")
            throughput.append(batch / (t1 - t0))
        avg_throughput = sum(throughput) / len(throughput)
        print(f"avg throughput: {avg_throughput:.3f}")

    if CAST:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            run()
    else:
        run()

if __name__ == "__main__":

    long_str = "This model is also a PyTorch torch.nn.Module subclass. \
                Use it as a regular PyTorch Module and refer to the PyTorch \
                documentation for all matter related to general usage and behavior."
    short_str = "Paris is the capital of [MASK]."
    text = [long_str] * batch
    
    CAST = 'cast' in sys.argv

    run_model('bert-base-uncased',
              BertTokenizerFast.from_pretrained('bert-base-uncased'),
              BertModel.from_pretrained('bert-base-uncased'),
              text, 10)
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     run_model('bert-base-uncased',
    #           BertTokenizerFast.from_pretrained('bert-base-uncased'),
    #           BertModel.from_pretrained('bert-base-uncased'),
    #           text, 2)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))

