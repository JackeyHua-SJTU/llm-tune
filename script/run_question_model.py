from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from QuestionModel import QuestionModel
import tqdm
import torch
import json


def train(questionModel, model, optim):

    epochs = 25

    for _ in tqdm.tqdm(range(epochs)):
        for X, a in questionModel:
            X = X.to(device)
            a = a.to(device)
            # optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            # print(f"loss is {loss}")
            loss.backward()
            optim.step()
            optim.zero_grad()
        torch.save(model.state_dict(), "model_state.pt")
        print(infer("A caterpillar changing into a butterfly is an example of"))

def infer(inp):
    inp = "<startofstring>" + inp + "<bot>:"
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_new_tokens=40, max_length=50)
    output = tokenizer.decode(output[0])
    return output

def validate():
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

    for i in range(100):
        print(infer(question[i]))
    

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = '/home/zdhua/gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token": "<pad>",
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

questionModel = QuestionModel("../data/train.jsonl", tokenizer)
questionModel =  DataLoader(questionModel, batch_size=64)

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(questionModel, model, optim)

print("infer from model : ")
# while True:
#   inp = input()
#   print(infer(inp))
validate()