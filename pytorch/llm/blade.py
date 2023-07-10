import sys
import time
import torch
from PIL import Image
from transformers import (BertTokenizerFast, BertModel, GPT2TokenizerFast, GPT2Model,
                          ViTImageProcessor, ViTModel, AutoImageProcessor, ResNetForImageClassification)
import torch_blade

DRYRUN = 'dryrun' in sys.argv
CAST = 'cast' in sys.argv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = 2048 if torch.cuda.is_available() else 256

@torch.no_grad()
def run_model(name, tokenizer, model, input, steps):
    if DRYRUN:
        return

    model.to(device)

    encoded_input = tokenizer(input, return_tensors='pt')
    encoded_input.to(device)
    # model = torch_blade.optimize(model, allow_tracing=True, model_inputs=(encoded_input['pixel_values'], ))
    # model = torch_blade.optimize(model, allow_tracing=True, model_inputs=(encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids'], ))

    model(**encoded_input)


    best_throughput = 0
    for _ in range(steps):
        t0 = time.time()
        # encoded_input = tokenizer(input, return_tensors='pt')
        # encoded_input.to(device)
        model(**encoded_input)
        t1 = time.time()
        cur_throughput = len(input) / (t1 - t0)
        best_throughput = cur_throughput if cur_throughput > best_throughput else best_throughput
        print(name + ' ' + str(cur_throughput))
    
    print(f"best throughput: {best_throughput}")


# image = Image.open('./cats.jpg')

# run_model('resnet-50',
#           AutoImageProcessor.from_pretrained("microsoft/resnet-50"),
#           ResNetForImageClassification.from_pretrained("microsoft/resnet-50"),
#           [image]*16, 10)

run_model('bert-base-uncased',
            BertTokenizerFast.from_pretrained('bert-base-uncased'),
            BertModel.from_pretrained('bert-base-uncased'),
            ['Paris is a very beautiful city, Paris is the capital of [MASK].']*batch, 10)


