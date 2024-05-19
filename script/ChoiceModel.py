from torch.utils.data import Dataset
import json

class ChoiceModel(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))
        self.question = []
        self.choice = []
        self.answer = []

        for i in self.data:
            self.question.append(i['question'])
            self.answer.append(f"The answer is {i['answerKey']}")
            temp_str = ""
            for idx, name in enumerate(i['choices']['label']):
                temp_str += name + '. ' + i['choices']['text'][idx] + ' '
            self.choice.append(temp_str)

        self.X = []
        for i in range(0, len(self.question)):
            self.X.append("<startofstring>" + self.question[i] + "<choice>:" + self.choice[i] + "<bot>:" + self.answer[i] + "<endofstring>")

        # self.X = self.X[:6000]

        print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=60, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])