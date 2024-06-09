import torch
import os
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import TextGenerationPipeline

# model = '/state/partition/wmhu/model/llama-2-7b-hf'
# # model = '/state/partition/wmhu/model/llama-2-13b-hf'

# # llama_model = AutoModelForCausalLM.from_pretrained(
# #     model,
# #     local_files_only=True,
# #     torch_dtype=torch.float16,
# #     quantization_config= BitsAndBytesConfig (
# #         # load_in_4bit=True,
# #         bnb_4bit_quant_type="nf4",
# #         bnb_4bit_use_double_quant=True,
# #         bnb_4bit_compute_dtype=torch.bfloat16
# #     ),
# #     device_map='auto'
# # )

# save_path = '/home/zdhua/temp_gpt/model'

# # llama_model.save_pretrained(save_path)

# tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
# examples = [
#     tokenizer(
#         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#     )
# ]

# quantize_config = BaseQuantizeConfig(
#     bits=4,  # 将模型量化为 4-bit 数值类型
#     group_size=128,  # 一般推荐将此参数的值设置为 128
#     desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
# )

# # 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
# model = AutoGPTQForCausalLM.from_pretrained(model, quantize_config)

# # 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
# model.quantize(examples)

# # 保存量化好的模型
# model.save_quantized(save_path)

model = AutoGPTQForCausalLM.from_quantized("./model", device="cpu")
tokenizer = AutoTokenizer.from_pretrained('/state/partition/wmhu/model/llama-2-7b-hf', use_fast=True)
# print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])