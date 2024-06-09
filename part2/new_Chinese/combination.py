import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer, GPT2Tokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

llama_tokenizer_dir = '/home/zdhua/gpt2-fine-tune/part2/new_Chinese/transformers_tokenizer/llama/tokenizer.model'
chinese_sp_model_file = "tokenizer.model"

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# with open(chinese_sp_model, 'r', encoding='utf-16') as f:  # Try 'gbk' encoding
#     chinese_tokens = f.readlines()

# chinese_tokens = [token.strip() for token in chinese_tokens if token.strip()]

# # Merge Chinese tokens into LLaMA tokenizer
# for token in chinese_tokens:
#     llama_tokenizer.add_tokens(token)

# # Save the merged tokenizer
# output_hf_dir = 'transformers_tokenizer/llama_chinese'
# os.makedirs(output_hf_dir, exist_ok=True)
# llama_tokenizer.save_pretrained(output_hf_dir)

# print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")
## Add Chinese tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New {len(llama_spm.pieces)}")

## Save
output_sp_dir = 'transformers_tokenizer/llama_chinese'
output_hf_dir = 'transformers_tokenizer/llama_chinese'  # the path to save Chinese-LLaMA tokenizer
with open(output_sp_dir + '/chinese_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Csaved to {output_hf_dir}")

# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
text = '''饮水思源,爱国荣校。Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.'''
print("Test text:\n", text)
print(f"LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Combined Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")