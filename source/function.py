import torch
import json

def loads_from_file(path):
    f = open(path, 'r')
    ret = json.loads(f.read())
    f.close()
    return ret

def get_input(slice, tokenizer, types):
    model_input = []
    mask_tensor = []
    pos = []
    ans = []
 
    for obj in slice:
        s = obj['left_context_token'] + ' <ent> ' + obj['mention_span'] + ' </ent> ' + obj['right_context_token']
        s = tokenizer.tokenize(s)
        pos.append(s.index('<ent>'))
        model_input.append(tokenizer.convert_tokens_to_ids(s))
        mask_tensor.append([1]*len(s))

        ans_tmp = [0.] * 130
        for ty in obj['y_str'] : 
            if ty in types:
                ans_tmp[types.index(ty)] = 1.
        ans.append(ans_tmp)
 
    maxlen = max([len(s) for s in model_input])
    model_input = [ (s+[0]*(maxlen-len(s))) for s in model_input]
    mask_tensor = [ (s+[0]*(maxlen-len(s))) for s in mask_tensor]
    
    model_input = torch.tensor(model_input)
    mask_tensor = torch.tensor(mask_tensor)
    ans = torch.tensor(ans)

    return model_input, mask_tensor, pos, ans