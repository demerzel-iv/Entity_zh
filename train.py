import torch
import json

from tqdm import tqdm
from random import randint
from transformers import BertTokenizer, BertModel

from source.model import noname
from source.parser import parse

def loads_from_file(path):
    f = open(path, 'r')
    ret = json.loads(f.read())
    f.close()
    return ret

def get_input(obj):
    text = obj['left_context_token'] + ' [ENL] ' + obj['mention_span'] + ' [ENR] ' + obj['right_context_token']
    text = tokenizer.tokenize(text)
    pos = text.index('[ENL]')
    text = tokenizer.convert_tokens_to_ids(text)
    text = torch.tensor([text])

    ans = [0.] * 130
    for ty in obj['y_str'] : 
        ans[types.index(ty)] = 1.
    ans = torch.tensor(ans)

    return text, pos, ans

def train():
    model.train()

    optim = torch.optim.Adam(params=model.parameters(), lr = 0.001)

    epoch_cnt = 0
    avg_loss = 0

    for epoch in tqdm(range(1000)):
        optim.zero_grad()

        loss = torch.tensor(0)

        for i in range(40):
            text, pos, ans = get_input(data[randint(0,len(data)-1)])

            text = text.to(config.dev)
            ans = ans.to(config.dev)
            out = model(text, pos)

            loss = loss - torch.sum(
                torch.log(out) * ans
                + torch.log(1-out) * (1-ans)
            )

        epoch_cnt += 1
        avg_loss += loss.item()

        if epoch_cnt % 200 == 0:
            print('avg_loss = ', avg_loss/200/17)
            avg_loss = 0

        loss.backward()
        optim.step()

def test():
    test_data = loads_from_file('./test_data.json')

    avg_loss = 0

    model.eval()

    for i in range(len(test_data)):
        text, pos, ans = get_input(test_data[i])
        text = text.to(config.dev)
        ans = ans.to(config.dev)
        out = model(text,pos)

        loss = - torch.sum(
            torch.log(out) * ans
            + torch.log(1-out) * (1-ans)
            )

        avg_loss += loss.item()

    #print(out)
    #print(ans)

    print("avg_loss on test : " , avg_loss/len(test_data))

def main():
    global tokenizer, types, data, model, config

    config = parse()

    #path = '/home/demerzel/Desktop/workshop/NLP/bert'
    path = '.'

    tokenizer = BertTokenizer.from_pretrained(path + '/vocab.txt', additional_special_tokens = ['[ENL]','[ENR]']) 
    types = loads_from_file(path + '/types.json')
    data = loads_from_file(path + '/train_data.json')
    model = noname().to(config.dev)

    train()

    test()

    torch.save(model,path + '/save.pth')


if __name__ == '__main__':
    main()
