import torch
import json

from tqdm import tqdm
from random import shuffle
from transformers import BertTokenizer, BertModel

from source.model import noname
from source.parser import parse
from source.count import counter
from source.function import get_input, loads_from_file

def train(epoch_num : int):
    model.train()
    cnter = counter()

    for i in range(epoch_num):
        print("\n===============Epoch.{}===============\n".format(i))

        optim = torch.optim.Adam(params=model.parameters(), lr = 0.001)
        avg_loss = 0
        batch_size = 40
        batch_num = int(len(data)/batch_size)
        shuffle(data)

        for batch_cnt in tqdm(range(batch_num)):

            optim.zero_grad()

            model_input, mask_tensor, pos, ans = get_input(data[batch_cnt * batch_size : (batch_cnt+1) * batch_size], tokenizer, types)

            model_input = model_input.to(config.dev)
            mask_tensor = mask_tensor.to(config.dev)
            ans = ans.to(config.dev)

            model_output = model(model_input, pos, mask_tensor)

            loss = - torch.sum(
                torch.log(model_output) * ans
                + torch.log(1-model_output) * (1-ans)
            )
            
            cnter.count(model_output.view(-1).tolist(),ans.view(-1).tolist())

            avg_loss += loss.item()

            loss.backward()
            optim.step()

            #if (batch_cnt+1) % 200 == 0:
        print('avg_loss = ', avg_loss/batch_num/batch_size)
        cnter.output()
        cnter.clear()



def test():
    test_data = loads_from_file(datapath + 'test_data.json')

    print('testing : ')

    model.eval()
    cnter = counter()

    model_input, mask_tensor, pos, ans = get_input(test_data, tokenizer, types)
    model_input = model_input.to(config.dev)
    mask_tensor = mask_tensor.to(config.dev)
    ans = ans.to(config.dev)

    model_output = model(model_input, pos, mask_tensor)
    loss = - torch.sum(
        torch.log(model_output) * ans
        + torch.log(1-model_output) * (1-ans)
    )
    
    cnter.count(model_output.view(-1).tolist(),ans.view(-1).tolist())

    print("avg_loss on test : " , loss.item()/len(test_data))
    cnter.output()

def main():
    global tokenizer, types, data, model, config, datapath

    config = parse()

    path = '.'
    datapath = '/home/demerzel/Desktop/workshop/NLP/data/'

    print('prework on data')
    data = []
    for line in tqdm(
        open(datapath + 'train_data_en.json','r').readlines() 
        + open(datapath + 'trans_data_zh.json','r').readlines()
        + open(datapath + 'train_data_zh.json','r').readlines()
        ):
        data.append(json.loads(line))

    #tokenizer = BertTokenizer.from_pretrained(path + '/vocab.txt', additional_special_tokens = ['[ENL]','[ENR]']) 
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer.add_special_tokens({'additional_special_tokens':["<ent>","</ent>"]})

    types = loads_from_file(path + '/types.json')
    model = noname(len(tokenizer)).to(config.dev)
#model = torch.load('./model/save.pth').to(config.dev)

    train(10)
    test()

    torch.save(model,path + '/model/save.pth')


if __name__ == '__main__':
    main()
