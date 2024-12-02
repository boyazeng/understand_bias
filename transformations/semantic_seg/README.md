## Semantic Segmentation

### Install environment
```
conda create -n bias_seg python=3.8 -y
conda activate bias_seg
conda install pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
git clone https://github.com/czczup/ViT-Adapter.git
cd ViT-Adapter/segmentation
ln -s ../detection/ops ./
cd ops
bash make.sh # compile deformable attention
```
move transform.py to ViT-Adapter/segmentation/ and download checkpoint

### Organize transform.py and download model
```
mv transform.py ViT-Adapter/segmentation/
cd ViT-Adapter/segmentation/
wget https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_ade20k.zip
unzip mask2former_beitv2_adapter_large_896_80k_ade20k.zip
```

### Perform transformation

Note you need to run the script under ViT-Adapter/segmentation/.

By default, we use a batch size of 1. This ensures that the model can inference on a GPU with 11GB memory. If you have a GPU with more memory, you can use a larger batch size. Also, for images with extremely large resolution, even using a batch size of 1 still can raise OOM error, but this happens very rarely (2 out of 1M images).

Set `dataset` to the name of the dataset in `data_path.py`.

```
python transform.py --dataset $dataset --split train --num 1_000_000
python transform.py --dataset $dataset --split val --num 10_000
```
