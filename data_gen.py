import numpy as np
import pandas as pd
from vocab import EXPR_TOKENIZER

def gen_equiv(num, partitions):
    if(partitions == 1):
        return str(num)
    pind = np.zeros(partitions+1, dtype=int)
    pind[1:-1] = np.random.randint(0, num, partitions-1)
    pind[-1] = num
    pind.sort()
    pdiffs = pind[1:] - pind[:-1]
    return "+".join(str(pd) for pd in pdiffs)

def sample_relation_text(max_num, max_partitions):
    num = np.random.randint(1, max_num)
    p1, p2 = tuple(np.random.randint(1, max_partitions, 2))
    s1 = gen_equiv(num, p1)
    s2 = gen_equiv(num, p2)
    return f"{s1}={s2}"

def sample_relation(max_num, max_partitions):
    num = np.random.randint(1, max_num)
    p1, p2 = tuple(np.random.randint(1, max_partitions, 2))
    s1 = gen_equiv(num, p1)
    s2 = gen_equiv(num, p2)
    return s1, s2

def gen_dataset(num_samples, max_num, max_partitions=3):
    relation_l = []
    for i in range(num_samples):
        relation_l.append(sample_relation(max_num, max_partitions))
    return relation_l

# relation_l = gen_dataset(10000, 10, 3) + gen_dataset(10000, 100, 3) + gen_dataset(1000, 5000, 3) + gen_dataset(100, 500000, 3)
# import pandas as pd
# dataset = pd.DataFrame(relation_l, columns=['EqA', 'EqB'])
# dataset.to_csv('data/strat_relation.csv', index=False)

# relation_l = gen_dataset(10000, 10, 3) + gen_dataset(100000, 100, 3) + gen_dataset(1000, 5000, 3) + gen_dataset(100, 500000, 3)
# eq_l = [f'{x}={y}' for x, y in relation_l]
# import pandas as pd
# dataset = pd.DataFrame(eq_l, columns=['Eq'])
# dataset.to_csv('data/equation_dataset.csv', index=False)

# print(gen_equiv(15, 4))
# print(sample_relation(10, 3))

import torch
from torch.utils import data
from torch.nn import Transformer
from vocab import PAD_IDX, BOS_IDX, EOS_IDX

def batch_expr_to_tensor(expr):
    batch_enc = EXPR_TOKENIZER.encode_batch(expr) #Tokenizes, adds padding to batch
    batch_id = [e.ids for e in batch_enc]
    batch_mask = [e.attention_mask for e in batch_enc] #Avoid processing padding w/ attention
    return torch.LongTensor(batch_id), ~torch.BoolTensor(batch_mask)

# function to collate data samples into batch tensors
def collate_strs(batch_strs):
    src_id, src_pad_mask = batch_expr_to_tensor(batch_strs) 
    return src_id, src_pad_mask

class ExpressionDataset(data.Dataset):
    def __init__(self, expr_l):
        self.expr_l = expr_l
    
    @classmethod
    def from_csv(cls, csv_fn='data/equation_dataset.csv'):
        df = pd.read_csv(csv_fn)
        return ExpressionDataset([expr for expr in df['Eq']])

    def __getitem__(self, index):
        return self.expr_l[index]

    def __len__(self):
        return len(self.expr_l)



