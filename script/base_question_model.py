from QuestionModel import QuestionModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import json

model_path = '/home/zdhua/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

data = []
with open("../data/test.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
question = []
correct = []
for i in data:
    question.append(i['question'])
    str = " and ".join(i['answers'])
    correct.append(str)


for i in range(50):
    prompt = "Question: " + question[i] + ", Answer: "
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    print(tokenizer.decode(output[0]))