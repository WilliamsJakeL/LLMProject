# Download the ARC Dataset
print("Downloading the ARC dataset")

import requests
import zipfile
import os

url = 'https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip' 
local_filename = "ARC-V1-Feb2018.zip"
# Download the file
r = requests.get(url, stream=True)
with open(local_filename, 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192): 
        f.write(chunk)
# Unzip the folder
with zipfile.ZipFile(local_filename, 'r') as zip_ref:
    zip_ref.extractall() 
# Delete the zip file
os.remove(local_filename)



## Tokenize the ARC Corpus
print("Tokenizing the ARC Corpus.")

# Modified from data/openwebtext/prepare.py
input_file_path = "ARC-V1-Feb2018-2/ARC_Corpus.txt"
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
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()




