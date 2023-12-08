# Select Kernal
Model run configurations for our 4 tests 

# Preparing OpenWeb data (first)
```
$ python data/openwebtext/prepare.py
```

The following configurations can be ran in parallel 

# Baseline (Dot Product)
```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --kernel_config=0 --out_dir=out-baseline
```

# Polynomial
```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --kernel_config=1 --out_dir=out-polynomial
```

# Periodic
```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --kernel_config=2 --out_dir=out-periodic
```

# Gaussian
```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --kernel_config=3 --out_dir=out-gaussian
```


## Evaluate GPT2 model through AI2 Reasoning Challenge (ARC)
<ol>
  <li> Download the ARC dataset and Tokenize the ARC Corpus. <br>
    <code> $ python data/arc/prepare.py </code><br>
  (Remark: if not properly work, download the ARC dataset from <a href="https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip">this link</a> and unzip the file at <code>data/arc</code> folder. Then rerun the script.)</li>
  <li> Rename the folder of each <code>ckpt.pt</code> to "out-arc-baseline", "out-arc-polynomial", "out-arc-periodic" or "out-arc-gaussian" respectively.</li>

  <li>Evaluate the models for each <b>kernel_config</b>:
    <ol>
      <li> Fine-tuning the GPT2 model.<br>
        <code> $ python train.py config/finetune_arc.py --init_from=resume --kernel_config=/// </code>
      </li>
      <li> Run evaluation.<br>
        <code> $ python eval_arc.py --kernel_config=/// </code> </li>
      </li>
      <li>
        Evaluate the next model.
      </li>
    </ol>
  </li>
</ol>


## Evaluate GPT2 model through EN-FR translation and BLEU
  <li>Evaluate the models one by one:
    <ol>
      <li> Fine-tuning the GPT2 model.<br>
        <code> $ python train.py config/finetune_en-fr.py --init_from=/// </code>
      </li>
      <li> Run evaluation.<br>
        <code> $ python eval_BLEU.py </code> </li>
      </li>
      <li>
        Evaluate the next model.
      </li>
    </ol>
  </li>
