import sys
import time
import torch
from PIL import Image
from transformers import (BertTokenizerFast, BertModel, GPT2TokenizerFast, GPT2Model,
                          ViTImageProcessor, ViTModel, AutoImageProcessor, ResNetForImageClassification)
from torchinfo import summary


text = ['Paris is the capital of [MASK].']

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

with torch.no_grad():
    encoded_input = tokenizer(text, return_tensors='pt')
    x = [encoded_input[k] for k in encoded_input.keys()]
    summary(model, input_data=x, depth=10)
