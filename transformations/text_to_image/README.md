## Text-to-Image

Please generate the captions first before this transformation.

### Install environment
```
(option 1: on top of the environment for caption generation)
conda activate bias_caption
conda install -c conda-forge diffusers

(option 2: create a new environment)
conda create -n bias_text2image
conda activate bias_text2image
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge diffusers transformers accelerate
pip install pillow scikit-image opencv-python
```


### Perform transformation
Set `dataset` to the name of the dataset in `data_path.py`.

To generate the training set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split train --num 1_000_000
```

To generate the validation set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split val --num 10_000
```
