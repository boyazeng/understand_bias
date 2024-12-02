## Variation Autoencoder

### Install latent diffusion
```
git clone https://github.com/CompVis/latent-diffusion latent_diffusion
cd latent_diffusion
conda env create -n bias_vae -f environment.yaml
conda activate bias_vae
pip install pytorch-lightning==1.6.1
wget https://ommer-lab.com/files/latent-diffusion/kl-f4.zip
unzip kl-f4.zip
cd ..
```

### Perform transformation
To transform the training set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split train --num 1_000_000
```

To transform the validation set:
```
python -m torch.distributed.launch --nproc_per_node=8 transform.py --dataset $dataset --split val --num 10_000
```
