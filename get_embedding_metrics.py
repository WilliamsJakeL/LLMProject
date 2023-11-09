## Script copied form jupyter notebook to do heavy lifting

import numpy as np
import sys; sys.path.append("../")
import tiktoken
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

train_data = np.memmap('data/openwebtext/train.bin', dtype=np.uint16, mode='r')

enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

model = GPT.from_pretrained("gpt2", {})

wte = None
for name, param in model.named_parameters():
    if name == 'transformer.wte.weight':
        wte = param

wte_embed = nn.Embedding(50257, 768)
wte_embed.weight = wte

## Find mean of embedding
batch_size = 10000
batches = math.floor(9035582489/batch_size)
m = 0
for i in tqdm(range(batches)):
    wte_OWT = wte_embed(torch.from_numpy((train_data[i*batch_size:(i+1)*batch_size]).astype(np.int64)))
    m = ((i*m) + torch.mean(wte_OWT))/(i+1)
#     m += 1
wte_OWT = wte_embed(torch.from_numpy((train_data[i*batch_size:]).astype(np.int64)))
m = ((batch_size*i*m) + torch.sum(wte_OWT))/len(train_data)
print(m)

## Find mean of embedding
std = 0
for i in tqdm(range(batches)):
    wte_OWT = wte_embed(torch.from_numpy((train_data[i*batch_size:(i+1)*batch_size]).astype(np.int64)))
    std = ((i*std) + torch.mean((wte_OWT-m)**2))/(i+1)
#     m += 1
wte_OWT = wte_embed(torch.from_numpy((train_data[i*batch_size:]).astype(np.int64)))
std = ((batch_size*i*m) + torch.sum((wte_OWT-m)**2))/len(train_data)
std = torch.sqrt(std).item()
print(std)