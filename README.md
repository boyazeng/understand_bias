# Understanding Bias in Large-Scale Visual Datasets

Official code for **Understanding Bias in Large-Scale Visual Datasets**

> [**Understanding Bias in Large-Scale Visual Datasets**](https://openreview.net/pdf?id=NGIIHlAEBt), NeurIPS 2024<br>
> [Boya Zeng*](https://boyazeng.github.io), [Yida Yin*](https://davidyyd.github.io), [Zhuang Liu](https://liuzhuang13.github.io) (*equal contribution)
> <br>University of Pennsylvania, UC Berkeley, Meta AI<br>
> [[`paper`](https://openreview.net/pdf?id=NGIIHlAEBt)][[`video`](https://www.youtube.com/watch?v=7cIaZmMhmZY)][[`slides`](https://neurips.cc/media/neurips-2024/Slides/95456.pdf)][[`project page`](https://boyazeng.github.io/understand_bias/)]

<p align="center">
<img src="./docs/static/images/teaser.png" width=50% height=50% 
class="center">
</p>

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.
To download the YFCC, CC, and Datacomp datasets, refer to [download_data/README.md](download_data/README.md).

Please update the paths to the datasets (``IMAGE_ROOTS``) and the paths to save the transformed images (``SAVE_ROOTS``) in [data_path.py](data_path.py).

## Transformations
<p align="center">
<img src="./docs/static/images/example_trans.png" width=80% height=80% 
class="center">
</p>

To perform transformations on images, follow the instructions in each subfolder under [transformations](transformations).

## Results

|transformation option|transformation|accuracy|
|:-|:-:|:-:|
|reference|Original images|82.0%|
|canny|Canny edge detection|71.0%|
|caption|LLaVA captioning|63.8% (short) / 66.1% (long)|
|depth|Depth estimation|73.1%|
|hog|Histogram of Oriented Gradients|79.0%|
|hpf|High-pass filter|79.2%|
|lpf|Low-pass filter|70.4%|
|mean_rgb|Mean RGB|48.5%|
|object_det|Object detection|61.9%|
|patch_shuffle|Patch shuffling|80.1% (random) / 81.2% (fixed)|
|pixel_shuffle|Pixel shuffling|52.2% (random) / 58.5% (fixed)|
|sam|Segment Anything Model|73.2%|
|seg|Semantic segmentation|69.8%|
|sift|Scale-Invariant Feature Transform|53.3%|
|text_to_image|Text-to-Image|58.1%|
|uncon_gen|Unconditional generation|77.6%|
|vae|Variational Autoencoder|77.4%|

## Dataset Classification
**Original images**

Replace ``$data_names`` with the names of the datasets you register in ``IMAGE_ROOTS`` of [data_path.py](data_path.py).

For example, if you register ``yfcc``, ``cc``, and ``datacomp`` in ``IMAGE_ROOTS``, you should set ``$data_names`` to ``yfcc,cc,datacomp``.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_tiny --epochs 30 --warmup_epochs 2 \
--batch_size 128 --lr 1e-3 --update_freq 4 \
--transformation reference --data_names $data_names \
--num_samples 1_000_000
```

**Transformed images**

Replace ``$transformation`` with the name of the transformation (``canny, depth, hog, hpf, lpf, mean_rgb, object_det, patch_shuffle, pixel_shuffle, sam, seg, sift, text_to_image, uncon_gen, vae``).
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_tiny --epochs 30 --warmup_epochs 2 \
--batch_size 128 --lr 1e-3 --update_freq 4 \
--transformation $transformation --data_names $data_names \
--num_samples 1_000_000
```
**Image captions**

Replace ``$caption_type`` with ``short`` or ``long``.

We search over the following hyperparameters:
- ``$epoch``: 1, 2, 4, 6
- ``$lr``: 1e-3, 1e-4, 1e-5.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model sentence-t5-base --epochs $epoch --warmup_steps $((epoch * 1562)) \
--weight_decay 0 --mixup 0 --cutmix 0 --smoothing 0 \
--batch_size 16 --lr $lr \
--transformation caption --caption_type $caption_type \
--data_names $name --num_samples 1_000_000
```

## Evaluate our pretrained checkpoint on original images

Download the checkpoint from [here](https://huggingface.co/spaces/boyazeng/understand_bias/resolve/main/original_ckpt.pth)
```
python main.py \
--transformation reference --data_names 'yfcc,cc,datacomp' \
--resume <path_to_checkpoint> --eval True
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) codebase.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@inproceedings{zengyin2024bias,
  title={Understanding Bias in Large-Scale Visual Datasets},
  author={Boya Zeng and Yida Yin and Zhuang Liu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```