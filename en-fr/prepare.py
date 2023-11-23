## Tokenize the ARC Corpus
print("Tokenizing the EN-FR Corpus.")


# Modified from data/openwebtext/prepare.py
input_file_path = "en-fr_40000.csv"
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import Dataset 

num_proc = 8
num_proc_load_dataset = num_proc

with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = [line.strip() for line in lines]
dataset = Dataset.from_dict({"text": data})

split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
print("Processing...")
def process(example):
    ids = enc.encode_ordinary(example['text']) 
    ids.append(enc.eot_token) 
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        if batch_idx>= len(dset):
            continue
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()




