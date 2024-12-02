## Depth

### Install environment
```
conda create -n bias_depth
conda activate bias_depth
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install gradio_imageslider gradio==4.29.0 matplotlib opencv-python
git clone https://github.com/DepthAnything/Depth-Anything-V2
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
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
