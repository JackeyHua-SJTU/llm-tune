from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from ChoiceModel import ChoiceModel
import tqdm
import torch
import json
import regex as re

def train(choiceModel, model, optim):

    epochs = 25

    for _ in tqdm.tqdm(range(epochs)):
        for X, a in choiceModel:
            # input id and attention mask
            X = X.to(device)
            a = a.to(device)
            # optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            # print(f"loss is {loss}")
            loss.backward() # bp
            optim.step()    # update
            optim.zero_grad()   # clear 
        torch.save(model.state_dict(), "model_state.pt")
        # ! Test correctness for every epoch
        print(infer("Sammy wanted to go to where the people were.  Where might he go?", "A. race track B. populated areas C. the desert D. apartment E. roadblock "))

def infer(que, cho):    # question + choice
    que = "<startofstring>" + que + "<choice>:"
    que += cho + "<bot>:"
    inp = tokenizer(que, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    # * generate the output
    output = model.generate(X, attention_mask=a, max_new_tokens=40, max_length=50)
    output = tokenizer.decode(output[0])
    return output

def validate():
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
    
    cnt = 0
    size = len(data)
    for i in range(len(question)):
        genAns = infer(question[i], choice[i])
        def parse(input:str):
            '''
            @return: Return the string between <bot>: and <endofstring>
            '''
            pattern = r'<bot>:\s*([^<]+)\s*<'
            match = re.findall(pattern, input)
            last_word = match[-1].strip()
            return last_word
        
        
        # ! in case ans is empty
        # * simply parse the last word, if it is not A/B/C/D/E, then it should be wrong
        try:
            ans = parse(genAns).split() # * Split and only get the LAST token !!! That is the possible choice.
            # print(f"the {i}th answer is {ans}")
            # print(f"the {i}th answer is {ans[-1]}")
            print(f"The {i}th question is {question[i]}, choice is {choice[i]}, correct ans is {answer[i]}, generated ans is {ans[-1]}")
            if ans[-1] == answer[i]:
                cnt += 1
        except:
            continue

    return cnt, cnt / size

device = "cuda" if torch.cuda.is_available() else "cpu"

# ! GPT2 model is downloaded to our server, running locally
model_path = '/home/zdhua/gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token": "<pad>",
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])    # * The answer that LLM should return
tokenizer.add_tokens(["<choice>:"]) # * The 5 choice of current question

model = GPT2LMHeadModel.from_pretrained(model_path)
# * Because we add new tokens, so we need to initialize weight for them
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

choice_model = ChoiceModel("../data/choice.json", tokenizer)
choice_model =  DataLoader(choice_model, batch_size=64)

model.train()   # ! set to training mode

# * optimizer used to update params
optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(choice_model, model, optim)

print("infer from model : ")

# ! input via command line
while True:
  que = input()
  cho = input()
  print(infer(que, cho))

# time, rate = validate()

# print(f"correct cnt is {time}, correct rate is {rate}")