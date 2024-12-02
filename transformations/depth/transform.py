import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from PIL import Image, ImageFile, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import sys
from pathlib import Path
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
sys.path.append(str(Path(__file__).resolve().parent / 'Depth-Anything-V2'))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils
from depth_anything_v2.dpt import DepthAnythingV2


class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['depth'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        path = self.paths[i]
        save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
        try:
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')

            if min(image.size) > 500:
                image = self.preprocess(image)
            return {"image": np.array(image), "save_path": save_path}
        except Exception as e:
            print(f"error to open {path} with error {e}")
            return {"image": None, "save_path": save_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default='cc')
    parser.add_argument('--split', type=str, choices=['val', 'train'], default='val')
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
    
    depth_anything = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    print(f'Loading checkpoints/depth_anything_v2_vitl.pth')
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_anything.to("cuda")
    depth_anything.eval()
    
    dataset = Image_Dataset(args)
    sampler_train = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed)
    data_loader = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=1,
        drop_last=False,
        num_workers=16,
        collate_fn=lambda x: x,
        shuffle=False,
    )
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Depth Anything {args.dataset} {args.split} split'
    for batch in metric_logger.log_every(data_loader, 10, header):
        sample = batch[0]
        try:
            depth = depth_anything.infer_image(sample["image"])
        except Exception as e:
            # save white image for images that can't be loaded
            print(f"error to process {sample['save_path']} with error {e}")
            continue
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        cv2.imwrite(sample["save_path"], depth)
        metric_logger.synchronize_between_processes()