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
    <code> $ python data/arc/prepare.py </code> </li>

  <li>Evaluate the models one by one:
    <ol>
      <li> Fine-tuning the GPT2 model.<br>
        <code> $ python train.py config/finetune_arc.py --init_from=/// --kernel_config=/// </code>
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
