import json
import torch
import random
import numpy as np
from multiprocessing import Pool
from zss import simple_distance, Node

def set_seed(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('seed:', seed)
set_seed()

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def from_postfix_to_tree(postfix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix:
        if p not in operators:
            st.append(Node(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(Node(p).addkid(b).addkid(a))
    return st.pop()

# train_data = load_data('../data/Math23K_train.jsonl')
# expr = set()
# for d in train_data:
#     expr.add(' '.join(d['postfix']))
# expr = list(expr)
# res = dict()
# for i in range(len(expr)):
#     for j in range(i, len(expr)):
#         tree1 = from_postfix_to_tree(expr[i].split(' '))
#         tree2 = from_postfix_to_tree(expr[j].split(' '))
#         tree_dis = simple_distance(tree1, tree2)
#         res[expr[i] + ' ; ' + expr[j]] = tree_dis
#         res[expr[j] + ' ; ' + expr[i]] = tree_dis
#     if i % 10 == 0:
#         print(i/len(expr))

# f = open('../data/tree_dis.json', 'w')  
# for d in res.items():
#     json.dump(d, f, ensure_ascii=False)
#     f.write("\n")
# f.close()