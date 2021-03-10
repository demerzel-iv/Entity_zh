import torch
import json
import random

from transformers import BertTokenizer, BertModel

def loads_from_file(path):
    f = open(path, 'r')
    ret = json.loads(f.read())
    f.close()
    return ret

def get_input(obj):
#text = obj['text'] 

    text = obj['left_context_token'] + ' [ENL] ' + obj['mention_span'] + ' [ENR] ' + obj['right_context_token']

    print('text = ', text)

    text = tokenizer.tokenize(text)
    pos = text.index('[ENL]')
    text = tokenizer.convert_tokens_to_ids(text)
    text = torch.tensor([text])

    return text, pos


global tokenizer


tokenizer = BertTokenizer.from_pretrained('./vocab.txt', additional_special_tokens = ['[ENL]','[ENR]']) 
types = loads_from_file('./types.json')
test_data = loads_from_file('./test_data.json')
model = torch.load('./model/save0.pth').to('cpu')

print("=========================")

for i in range(10):
    x = test_data[random.randint(0,100)]
    text, pos = get_input(x)
    out = model(text, pos)
    out = out.tolist()

    for i in range(len(out)):
        if(out[i] > 0.1):
            print(types[i] , ' , score = ' , out[i])

    print('===========================')
