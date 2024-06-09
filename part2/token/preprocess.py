import os
import json
import gzip
import gc
from tokenizers import Tokenizer

# 流程 先预处理数据文件，然后训练一个分词器，然后测试一下在训练前后的Token编码方式，然后在原来的LLM上跑结果，对比输出的回答
# 根据不同的LLM选择不同的原始分词器

def peek():
    with gzip.open("./data/c4-train.00000-of-01024.json.gz", "r") as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        json_list = json_str[1:-2].split('}\n{')
        # print(json_list)
        for i, data in enumerate(json_list):
            if i % 10000 == 0: print(i)
            data = json.loads('{'+data+'}')
            print(data)
            if 'text' not in data or 'timestamp' not in data or 'url' not in data:
                print(data)


def data_preprocess():
    dir = '/home/zdhua/temp_gpt/data'
    dataset = os.listdir(dir)
    files = list(filter(lambda x: '.json.gz' in x, dataset))
    print(files)
    for c4_data in files:
        with gzip.open(os.path.join(dir, c4_data), "r") as f:
            with open(os.path.join(dir, c4_data.replace(".json.gz", ".txt")), "w") as result:
                json_bytes = f.read()
                json_str = json_bytes.decode('utf-8')
                json_list = json_str[1:-2].split('}\n{')
                placeholder_cnt = 0
                for i, data in enumerate(json_list):
                    data = json.loads('{'+data+'}')
                    if 'text' not in data or 'timestamp' not in data or 'url' not in data:
                        print(data)
                        continue
                    if 'placeholder page' in data['text']:
                        placeholder_cnt += 1
                        continue
                    result.write(data['text'] + '\r\n')
                print(c4_data, "placeholder_knt:", placeholder_cnt)
        

if __name__ == '__main__':
    # peek()
    data_preprocess()