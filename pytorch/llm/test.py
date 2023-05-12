import time
import os
import torch
from transformers import pipeline, BertTokenizer , BertForMaskedLM


pipe = pipeline('fill-mask', model='bert-base-uncased')

print(pipe.model)
