import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.optim import Adam
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from tokenization import ChineseTokenizer

config = AutoConfig.from_pretrained("/home/zdhua/gpt2-fine-tune/new_Chinese/transformers_tokenizer/llama_chinese/tokenizer_config.json")
tokenizer = LlamaTokenizer.from_pretrained("/home/zdhua/gpt2-fine-tune/new_Chinese/transformers_tokenizer/llama_chinese/tokenizer.model")
# model = LlamaForCausalLM.from_pretrained('openbmb/MiniCPM-2B-sft-bf16-llama-format', config=config)
model = LlamaForCausalLM.from_pretrained('/state/partition/wmhu/model/llama-2-7b-hf', config=config)
model_vocab_size = model.get_output_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))

input_text = "你好"

# 将输入文本转换为 token
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 使用模型生成下一个词
output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("生成的文本：", generated_text)