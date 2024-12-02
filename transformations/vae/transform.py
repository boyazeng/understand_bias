import argparse
import glob
import numpy as np
import os
import sys
import torch
from latent_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms, get_image_backend
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from PIL import Image, ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("vae") + 1], "latent_diffusion"))
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("vae") + 1], "latent_diffusion/src"))
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("vae") + 1], "latent_diffusion/src/taming-transformers"))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


def to_pil(tensor):
    return ToPILImage()(tensor)

class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['vae'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, interpolation=Image.BICUBIC)
        self.transform = Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            ToTensor(),
        ])

    def __getitem__(self, index):
        path = self.paths[index]
        save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            # Add preprocessing to make sure that data are handled the same way for all datasets
            if min(img.size) > 500:
                img = self.preprocess(img)
            width, height = img.size[:2]
            img = self.transform(img)
            return {"image": ((img - 0.5) * 2), "height": height, "width": width, 'save_path': save_path}
        except:
            return {"image": None, "height": None, "width": None, 'save_path': save_path}
        

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num', type=int, default=None)
    args = parser.parse_args()
    
    utils.init_distributed_mode(args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    print(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    config = OmegaConf.load("./latent_diffusion/configs/autoencoder/autoencoder_kl_64x64x3.yaml")
    state_dict = torch.load("./latent_diffusion/model.ckpt", map_location="cpu")["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(state_dict, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to('cuda')
    model.eval()

    dataset = Image_Dataset(args)
    sampler_train = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed)
    data_loader = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=lambda x: x,
        shuffle=False,
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'VAE {args.dataset} {args.split} split'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            valid_indices = []
            save_paths = []
            images = []
            for i, each in enumerate(batch):
                if each['image'] is None:
                    continue
                
                valid_indices.append(i)
                save_paths.append(each['save_path'])
                images.append(each['image'])

            if len(valid_indices) == 0:
                continue

            images = torch.stack(images, dim=0)
            images = images.to('cuda')
            output, _ = model(images, sample_posterior=False)
            reconstructions = (output / 2) + 0.5
            reconstructions = torch.clamp(reconstructions, 0, 1)

            for i in range(reconstructions.size(0)):
                reconstructed = to_pil(reconstructions[i].cpu())
                reconstructed.save(save_paths[i])

            metric_logger.synchronize_between_processes()
