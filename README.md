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

