## Segment Anything Model (SAM)

### Install environment
```
conda create -n bias_sam -y
conda activate bias_sam
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python
```
Please adjust the pytorch installation command according to your CUDA version. https://pytorch.org/get-started/locally/

### Download the SAM model
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

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
