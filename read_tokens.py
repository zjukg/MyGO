import torch

from merge_tokens import get_entity_visual_tokens

if __name__ == "__main__":
    max_num = 8
    tokens, _ = get_entity_visual_tokens("DB15K", max_num=8)
    pairs = [[1910, 1912], [6843, 714], [1606, 3459], [5327, 7806]]
    for [a, b] in pairs:
        tokena = list(tokens[a])
        tokenb = list(tokens[b])
        for i in range(max_num):
            for j in range(max_num):
                if tokena[i] == tokenb[j]:
                    print(i, j, end=' ')
        print()