import torch
import numpy as np
from copy import deepcopy
from transformers import LlamaForCausalLM, LlamaTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symmetric_quantize(X: torch.Tensor):
    # Map the max to 127
    scaling_factor = 127 / torch.max(torch.abs(X))
    quantized = scaling_factor * X
    quantized = quantized.round()
    dequantized = quantized / scaling_factor
    return quantized.to(torch.int8), dequantized

def asymmetric_quantize(X: torch.Tensor):
    scaling_factor = 255 / torch.max(X) - torch.min(X)
    zeropoint = -(scaling_factor * torch.min(X)).round() - 128
    quantized = (scaling_factor * X + zeropoint).round()
    dequantized = (quantized - zeropoint) / scaling_factor
    return quantized.to(torch.int8), dequantized

def validate(model, tokenizer, input_text, max_length=50):
    input = tokenizer(input_text, return_tensors="pt")
    id = input["input_ids"].to(device)
    am = input["attention_mask"].to(device)
    output = model.generate(id, attention_mask=am, do_sample=True, top_k=30, max_new_tokens=40, max_length=max_length)
    return tokenizer.decode(output[0])

def calc_ppl(model, tokenizer, text):
    input = tokenizer(text, return_tensors="pt")

    id = input["input_ids"].to(device)
    backup = id.clone()

    with torch.no_grad():
        output = model(id, labels=backup)

    p = output.loss

    return torch.exp(p)

def llama_quant():
    model_path = '/state/partition/wmhu/model/llama-2-7b-hf'
    # model = LlamaForCausalLM.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained("../gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("../gpt2")
    init_weights = [param.data.clone() for param in model.parameters()]
    model_abs = deepcopy(model).to(device)
    weight_abs = []
    for param in model_abs.parameters():
        _, dequant = symmetric_quantize(param.data)
        param.data = dequant
        weight_abs.append(dequant)

    model_zp = deepcopy(model).to(device)
    weight_zp = []

    for param in model_zp.parameters():
        _, dequant = asymmetric_quantize(param.data)
        param.data = dequant
        weight_zp.append(dequant)


    text = "how are you?"
    rnd = 10
    ppl_init = 0
    ppl_abs = 0
    ppl_zp = 0
    for _ in range(rnd):
        answer_init = validate(model, tokenizer, text)
        answer_abs = validate(model_abs, tokenizer, text)
        answer_zp = validate(model_zp, tokenizer, text)
        # print(f"answer from init is {answer_init}, \nfrom abs quant model is {answer_abs}, \nfrom zp quant model is {answer_zp}")
        ppl_init += calc_ppl(model, tokenizer, answer_init)
        ppl_abs += calc_ppl(model, tokenizer, answer_abs)
        ppl_zp += calc_ppl(model, tokenizer, answer_zp)
    print(f"ppl init is {ppl_init / rnd}")
    print(f"ppl abs is {ppl_abs / rnd}")
    print(f"ppl zp is {ppl_zp / rnd}")

def int8_quant():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int8_model = AutoModelForCausalLM.from_pretrained("../gpt2", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("../gpt2")

    text = "how are you?"
    rnd = 10
    ppl = 0

    for _ in range(rnd):
        ans = validate(int8_model, tokenizer, text)

        ppl += calc_ppl(int8_model, tokenizer, ans)
    print(f"ppl is {ppl / rnd}")

if __name__ == '__main__':
    # arr = [256, 8445, 412, 3543]
    # arr = torch.tensor(arr)
    # q, dq = symetric_quantize(arr)
    # q1, dq1 = zeropoint_quantize(arr)
    # print(f"abs quantize : quantized is {q}, dequantized is {dq}")
    # print(f"zeropoint quantize : quantized is {q1}, dequantized is {dq1}")
    llama_quant()
    int8_quant()