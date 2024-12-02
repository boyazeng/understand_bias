## Unconditional Generation

### Install environment
```
conda env create -n bias_uncondgen -f environment.yaml
conda activate bias_uncondgen
```

### Perform transformation
Note by default, we train a `DiT-B/2` model with max steps of 275_000, a global batch size of 1024 to generate images of size 256x256. This can fit on a single node of 8GPUs, each with 11GB memory. If you have GPUs with more memory, you can train a larger model (`DiT-L/2`, `DiT-XL/2`), use a larger batch size, and/or use a larger input size (512x512).

Set `dataset` to the name of the dataset in `data_path.py`.

To train a DiT model on one dataset: 
```
accelerate launch --multi_gpu --num_processes 8 train.py \
--dataset $dataset --num 1_000_000
```

To generate the training set:
```
python -m torch.distributed.launch --nproc_per_node=8 sample.py \
--ckpt $ckpt --dataset $dataset --split train --num 1_000_000
```

To generate the validation set:
```
python -m torch.distributed.launch --nproc_per_node=8 sample.py \
--ckpt $ckpt --dataset $dataset --split val --num 10_000
```