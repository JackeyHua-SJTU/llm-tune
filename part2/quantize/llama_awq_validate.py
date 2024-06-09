from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

def calc_ppl(model, tokenizer, text):
    input = tokenizer(text, return_tensors="pt")

    id = input["input_ids"].to("cuda")
    backup = id.clone()

    with torch.no_grad():
        output = model(id, labels=backup)

    p = output.loss

    return torch.exp(p)

model_name = './llama_awq'

model = AutoAWQForCausalLM.from_pretrained(model_name, safetensors=True, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

while True:
    print("input your prompt: ")
    prompt = input()

    tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.to("cuda")

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
    )

    output = tokenizer.decode(generation_output[0])

    print("Output: ", tokenizer.decode(generation_output[0]))
    print("ppl : ", calc_ppl(model, tokenizer, output))