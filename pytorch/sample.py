from transformers import pipeline

# print(pipeline('sentiment-analysis')('I love you'))

generator = pipeline(model="gpt2")


