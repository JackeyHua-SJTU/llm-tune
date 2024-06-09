import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers import decoders, Regex
from tokenizers.models import BPE, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch


def zh_tokenization():
    Tokenizer()
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
    ])
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], 
                         vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel().alphabet())
    file_path = "/home/zdhua/temp_gpt/data/"
    dataset = [f"{file_path}sentences.txt"]
    tokenizer.train(files=dataset, trainer=trainer)
    save_path = "/home/zdhua/temp_gpt/zh_tokenizer.json"
    tokenizer.save(save_path)
    output = tokenizer.encode("Hello, y'all! How are you 我?\nwhat is the meal today")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def bpe_tokenization():
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
    ])
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"], 
                         vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel().alphabet())
    file_path = "/home/zdhua/temp_gpt/data/"
    dataset = [f"{file_path}c4-train.0000{i}-of-01024.txt" for i in range(10)]
    for i in range(10, 16):
        dataset.append(f"{file_path}c4-train.000{i}-of-01024.txt")
    tokenizer.train(files=dataset, trainer=trainer)
    save_path = "/home/zdhua/temp_gpt/tokenizer.json"
    tokenizer.save(save_path)
    output = tokenizer.encode("Hello, y'all! How are you 您?\nwhat is the meal today")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def wordpiece_tokenization():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
    ])
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]", "<s>", "</s>"], 
                         vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel().alphabet())
    file_path = "/home/zdhua/temp_gpt/data/"
    dataset = [f"{file_path}c4-train.0000{i}-of-01024.txt" for i in range(10)]
    for i in range(10, 16):
        dataset.append(f"{file_path}c4-train.000{i}-of-01024.txt")
    tokenizer.train(files=dataset, trainer=trainer)
    save_path = "/home/zdhua/temp_gpt/wp_tokenizer.json"
    tokenizer.save(save_path)
    output = tokenizer.encode("Hello, y'all! How are you 您?\nwhat is the meal today")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def wordlevel_tokenization():
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex(' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
    ])
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[MASK]", "<s>", "</s>"], 
                         vocab_size=32000,
                         initial_alphabet=pre_tokenizers.ByteLevel().alphabet())
    file_path = "/home/zdhua/temp_gpt/data/"
    dataset = [f"{file_path}c4-train.0000{i}-of-01024.txt" for i in range(10)]
    for i in range(10, 16):
        dataset.append(f"{file_path}c4-train.000{i}-of-01024.txt")
    tokenizer.train(files=dataset, trainer=trainer)
    save_path = "/home/zdhua/temp_gpt/wl_tokenizer.json"
    tokenizer.save(save_path)
    output = tokenizer.encode("Hello, y'all! How are you 您?\nwhat is the meal today")
    print(output.tokens)
    output = tokenizer.decode(output.ids)
    print(output)

def validate():
    file = "/home/zdhua/temp_gpt/wl_tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file, eos_token='</s>')
    while (True):
        query = input()
        output = tokenizer([query], max_length=10, truncation=False, stride=2, return_overflowing_tokens=True, return_length=True)
        # output = tokenizer.encode(query)
        print(output.tokens)
        # output = tokenizer.decode(output.ids)
        print(output)

def llama_inference():
    model_path = '/state/partition/wmhu/model/llama-2-7b-hf'
    model = LlamaForCausalLM.from_pretrained(model_path)
    file = "/home/zdhua/temp_gpt/bpe_tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file, eos_token='</s>')
    official_tokenizer = LlamaTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    while(True):
        print("Your text plz: ")
        input_text = input()

        encoding = tokenizer(input_text, return_tensors="pt")
        X = encoding["input_ids"].to(device)
        a = encoding["attention_mask"].to(device)
        # * generate the output
        output = model.generate(X, attention_mask=a, max_new_tokens=40, max_length=50)
        output = tokenizer.decode(output[0])

        off = official_tokenizer(input_text, return_tensors='pt')
        X_prime = off["input_ids"].to(device)
        a_prime = off["attention_mask"].to(device)
        output_off = model.generate(X_prime, attention_mask=a_prime, max_new_tokens=40, max_length=50)
        output_off = official_tokenizer.decode(output_off[0])

        print(f"\033[91mOutput from our model is\033[0m {output}\n \033[91mOutput from official mode \033[0mis {output_off}")


if __name__ == '__main__':
    # bpe_tokenization()
    # zh_tokenization()
    # validate()
    llama_inference()
    # wordpiece_tokenization()
    # wordlevel_tokenization()