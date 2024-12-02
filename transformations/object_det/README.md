## Object Detection

### Install environment
```
conda create -n bias_det python=3.8 -y
conda activate bias_det
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install opencv-python
```

### Download model
```
wget https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_h/332552778/model_final_11bbb7.pkl && mv model_final_11bbb7.pkl VitDet_huge_LVIS.pkl
```
### Perform transformation

By default, we use a batch size of 1. This ensures that the model can perform inference on a GPU with 11GB memory. Please feel free to use a larger batch size if you have a GPU with more memory.

Set `dataset` to the name of the dataset in `data_path.py`.

To transform the training set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split train --num 1_000_000
```

To transform the validation set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split val --num 10_000
```
