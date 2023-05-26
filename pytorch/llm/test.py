import time
import os
import torch
import argparse
from transformers import BertTokenizerFast, BertForMaskedLM, GPT2Model, GPT2TokenizerFast

def main():
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model = GPT2Model.from_pretrained('gpt2')

    print(model)

    # for _ in range(10):
    #     model(**encoded_input)

if __name__ == "__main__":
    main()
