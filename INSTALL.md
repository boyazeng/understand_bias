# Installation

We provide installation instructions for all classification experiments here.

## Dependency Setup
```
conda create -n understand_bias python=3.8 -y
conda activate understand_bias
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python timm==0.6.12 tensorboardX==2.6.2.2 six==1.16.0 pillow==10.3.0 scikit-image==0.21.0 numpy==1.23.1 sentence-transformers pandas pyarrow fastparquet
```