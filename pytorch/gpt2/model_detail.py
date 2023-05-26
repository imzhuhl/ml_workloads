import time
import os
import torch
import torch.nn as nn
import argparse
from torchinfo import summary
from transformers import GPT2Model, GPT2TokenizerFast

"""
torch.addmm([2304], [10, 784], [768, 2304])

"""

def test_embedding():
    embedding = nn.Embedding(50000, 784)
    input = torch.LongTensor([1 for _ in range(1000)])
    for i in range(100000000):
        embedding(input)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2, help="batch_size")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    text = ["Once upon a time," for _ in range(args.batch)]
    model = GPT2Model.from_pretrained('gpt2')

    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)
    
    # summary(model, input_data={"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"]})
    pred = model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])


    # for _ in range(10):
    #     model(**encoded_input)

if __name__ == "__main__":
    # main()
    test_embedding()
