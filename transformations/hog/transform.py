import argparse
import cv2
import glob
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pdb
import random
from skimage.feature import hog
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, PngImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['hog'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        # we resize all images to have short edge of 500, so we omit the preprocess step that resize large images to have short edge of 500
        self.transform = transforms.Resize(500, transforms.InterpolationMode.BICUBIC)
    
    def hog_transform(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, hog_image = hog(image_gray, visualize=True, transform_sqrt=True)
        hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return hog_image
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        try:
            path = self.paths[i]
            save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = self.transform(img)
            hog_image = self.hog_transform(np.array(img))
            cv2.imwrite(save_path, hog_image)
        except Exception as e:
            print(f"error to open {path} with error {e}")


if __name__ == "__main__":
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
    
    dataset = Image_Dataset(args)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    # use torch dataloader to do parallel processing
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=16, collate_fn=lambda x: x)
    torch.distributed.barrier()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'HOG {args.dataset} {args.split} split'
    for batch in metric_logger.log_every(dataloader, 10, header):
        metric_logger.synchronize_between_processes()
