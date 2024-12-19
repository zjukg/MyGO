import json
import torch

from collections import Counter, defaultdict

def load_ent_map(dataset):
    ent_map = {}
    if dataset == "FB15K-237" or dataset == "DB15K" or dataset == "WN9":
        f = open("data/{}/entities.txt".format(dataset), "r")
        for line in f.readlines():
            ent = line.replace('\n', '')
            ent_map[ent] = ent
    else:
        f = open("data/{}/entity2id.txt".format(dataset), "r")
        for line in f.readlines():
            ent, id = line[:-1].split(' ')
            ent_map[ent] = int(id)
    return ent_map


def get_entity_visual_tokens(dataset, max_num, type="beit"):
    if type == "beit":
        tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
        token_size = 8192
    elif type == 'vqgan':
        tokenized_result = json.load(open("tokens/{}-visual-vqgan.json".format(dataset), "r"))
        token_size = 1024
    else:
        raise NotImplementedError
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(token_size + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        key = str(i) if dataset == "DB15K" else i
        if entity_to_token[key] != []:
            entid_tokens.append(entity_to_token[key])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([token_size - 1] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_with_limit_db15k(dataset, max_num, max_img=None):
    assert max_img is not None
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for id in tokenized_result.keys():
        tokenized_result[id] = tokenized_result[id][0: min(len(tokenized_result[id]), 196 * max_img)]
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[str(i)] != []:
            entid_tokens.append(entity_to_token[str(i)])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_with_limit(dataset, max_num, max_img=None):
    if dataset == "DB15K":
        return get_entity_visual_tokens_with_limit_db15k(dataset, max_num, max_img)
    assert max_img is not None
    tokenized_result = json.load(open("tokens/{}-visual-v2.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for id in tokenized_result.keys():
        tokenized_result[id] = tokenized_result[id][0: min(len(tokenized_result[id]), 196 * max_img)]
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[i] != []:
            entid_tokens.append(entity_to_token[i])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_textual_tokens_db15K(dataset, max_num, type):
    if type == "bert":
        tokenized_result = json.load(open("tokens/{}-textual-v2.json".format(dataset), "r"))
    elif type == "roberta":
        tokenized_result = json.load(open("tokens/{}-textual-roberta.json".format(dataset), "r"))
    elif type == "llama":
        tokenized_result = json.load(open("tokens/{}-textual-llama.json".format(dataset), "r"))
    else:
        raise NotImplementedError
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[str(i)] == max_num:
            entid_tokens.append(entity_to_token[str(i)])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[str(i)]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()

def get_entity_textual_tokens(dataset, max_num, type="bert"):
    if dataset == "DB15K":
        return get_entity_textual_tokens_db15K(dataset, max_num, type)
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[i] == max_num:
            entid_tokens.append(entity_to_token[i])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[i]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_textual_tokens_FB15K237(dataset, max_num):
    assert dataset == "FB15K-237"
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in entity_dict:
        if entity_to_token[i] == max_num:
            entid_tokens.append(entity_to_token[ent])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[i]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_FB15K237(dataset, max_num):
    assert dataset == "FB15K-237"
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in entity_dict:
        key = i
        if entity_to_token[key] != []:
            entid_tokens.append(entity_to_token[key])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


if __name__ == "__main__":
    dataset = "FB15K-237"
    max_token_num = 8
    a, b = get_entity_visual_tokens_FB15K237(dataset, max_token_num)
    print(a)
    
    