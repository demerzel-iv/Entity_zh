import torch
import json
import random

from transformers import BertTokenizer, BertModel


def get_input(text):
    print('text = ', text)

    text = tokenizer.tokenize(text)
    pos = [text.index('<ent>')]
    text = tokenizer.convert_tokens_to_ids(text)
    text = torch.tensor([text])

    return text, pos


global tokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer.add_special_tokens({'additional_special_tokens':["<ent>","</ent>"]})

types = json.loads(open('./types.json', 'r').read())
model = torch.load('./model/save.pth').to('cpu')

data = json.loads(open('./manual.json').read())

for x in data:
    text, pos = get_input(x)
    out = model(text, pos)
    out = out.view(-1).tolist()

    for i in range(len(out)):
        if(out[i] > 0.1):
            print(types[i] , ' , score = ' , out[i])

    print('===========================')
