## Script to evaluate model performance on the MMLU Benchmark (https://github.com/hendrycks/test)

"""
Sample from a trained model
"""
import os
import numpy as np
import pandas as pd
import time
import pickle
from contextlib import nullcontext
import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-baseline' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

EXP_NAME = 'Baseline'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

args = {'ntrain': 2, 'data_dir': 'data/MMLU/data/'}

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Copy some code in from MMLU test GitHub
choices = ["A", "B", "C", "D"]

def crop_prompt(prompt: str):
    global enc

    cropped_prompt = decode(encode(prompt)[:model.config.block_size])
    return cropped_prompt

def crop(s):
    prompt = crop_prompt(s)
    return prompt

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval_subj(args, subject, model, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args['ntrain']
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        prompt_ids = encode(prompt)
        x = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]) 
        c = model.get_next_token_options(x)

        # while True:
        #     try:
        #         c = openai.Completion.create(
        #             engine=engine,
        #             prompt=prompt,
        #             max_tokens=1,
        #             logprobs=100,
        #             temperature=0,
        #             echo=True
        #         )
        #         break
        #     except:
        #         print("pausing")
        #         time.sleep(1)
        #         continue
        lprobs = []
        for ans in answers:
            try:
                # lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
                lprobs.append(c[0,encode(ans)].item())
            except Exception as e:
                print("Warning: {} not found with error {}. Artificially adding log prob of -100.".format(ans, e))
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

# Get subjects
subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args['data_dir'], "test")) if "_test.csv" in f])
print("Subjects to test:", subjects)
# Take the test for each subject
all_cors = []
all_acc = []
all_probs = []
all_subjects = []
with torch.no_grad():
    with ctx:
        for i, subject in enumerate(subjects):
            dev_df = pd.read_csv(os.path.join(args['data_dir'], "dev", subject + "_dev.csv"), header=None)[:args['ntrain']]
            test_df = pd.read_csv(os.path.join(args['data_dir'], "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval_subj(args, subject, model, dev_df, test_df)
            all_cors += [cors]
            all_acc += [acc]
            all_probs += [probs]
            all_subjects += [subject]

pickle.dump({'Subject': all_subjects, 'Correctness': all_cors, "Accuracy": all_acc, 'All Probabilities': all_probs}, open(EXP_NAME+".pickle", 'wb'))