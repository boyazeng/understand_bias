## Canny edge detector

### Install environment
Please use the same environment for dataset classification.

### Perform transformation

Set `dataset` to the name of the dataset in `data_path.py`.

To transform the training set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split train --num 1_000_000
```

To transform the validation set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split val --num 10_000
```