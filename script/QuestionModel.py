from torch.utils.data import Dataset
import json

class QuestionModel(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = []
        with open(path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.deal = []
        for i in self.data:
            self.deal.append(i['question'])
            str = " and ".join(i['answers'])
            self.deal.append(str)
        self.X = []
        for i in range(0, len(self.deal), 2):
            self.X.append("<startofstring>" + self.deal[i] + "<bot>:" + self.deal[i + 1] + "<endofstring>")

        # self.X = self.X[:6000]
        print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=60, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])