from ChoiceModel import ChoiceModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import json

model_path = '/home/zdhua/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

data = json.load(open("../data/test_choice.json", "r"))
question = []
choice = []
answer = []
for i in data:
    question.append(i['question'])
    answer.append(i['answerKey'])
    temp_str = ""
    for idx, name in enumerate(i['choices']['label']):
        temp_str += name + '. ' + i['choices']['text'][idx] + ' '
    choice.append(temp_str)

for i in range(50):
    prompt = "Question: " + question[i] + ", Choices: " + choice[i] + ", Answer: "
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    print(tokenizer.decode(output[0]))