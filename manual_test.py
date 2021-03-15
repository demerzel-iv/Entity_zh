import torch
import json
import random

from transformers import BertTokenizer, BertModel


def get_input(text):

    #text = obj['left_context_token'] + ' [ENL] ' + obj['mention_span'] + ' [ENR] ' + obj['right_context_token']

    print('text = ', text)

    text = tokenizer.tokenize(text)
    pos = text.index('[ENL]')
    text = tokenizer.convert_tokens_to_ids(text)
    text = torch.tensor([text])

    return text, pos


global tokenizer


tokenizer = BertTokenizer.from_pretrained('./vocab.txt', additional_special_tokens = ['[ENL]','[ENR]']) 
types = json.loads(open('./types.json', 'r').read())
model = torch.load('./model/save.pth').to('cpu')

data = open('./manual.json')

for x in data:
    text, pos = get_input(x)
    out = model(text, pos)
    out = out.tolist()

    for i in range(len(out)):
        if(out[i] > 0.1):
            print(types[i] , ' , score = ' , out[i])

    print('===========================')
