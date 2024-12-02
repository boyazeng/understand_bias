import argparse
import glob
import numpy as np
import os
import random
import shutil
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms, get_image_backend

from PIL import Image, PngImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils

def merge_anns(anns, h, w):
    """
    SAM generates potentially overlapping masks, so we need to merge them.
    For each pixel, we assign the object ID of the mask that covers it with the highest "predicted_iou".
    """
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    img = np.zeros((h, w))
    for idx, ann in enumerate(sorted_anns):
        img[ann['segmentation']] = idx+1
    return img

def contour(input_map):
    """
    Based on the per-pixel object ID map, we generate a binary map indicating pixel boundaries.
    If a pixel has one of its 4 connected neighbors with a different object ID, we mark it as a boundary pixel.
    """
    H, W = input_map.shape
    binary_map = np.zeros((H, W), dtype=int)
    
    original = input_map[1:-1, 1:-1]
    up = input_map[:-2, 1:-1]
    down = input_map[2:, 1:-1]
    left = input_map[1:-1, :-2]
    right = input_map[1:-1, 2:]

    binary_map[1:-1, 1:-1] = ((original != up) |
                              (original != down) |
                              (original != left) |
                              (original != right)).astype(int)
    
    return binary_map

class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['sam'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, interpolation=transforms.InterpolationMode.BICUBIC)

    def __getitem__(self, index):
        path = self.paths[index]
        save_path = os.path.join(self.output_dir, os.path.basename(path).replace(".jpg", ".png").replace(".jpeg", ".png"))
        try:
            with open(path, 'rb') as f:
                image = Image.open(f).convert('RGB')
            
            if min(image.size) > 500:
                image = self.preprocess(image)
            width, height = image.size[:2]
            return {"image": np.array(image), "height": height, "width": width, "save_path": save_path}
        except Exception as e:
            print(f"error to open {path} with error {e}")
            return {"image": None, "height": None, "width": None, "save_path": save_path}

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=1)
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

    sam = sam_model_registry['vit_l'](checkpoint='sam_vit_l_0b3195.pth')
    sam.to("cuda")
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
    )

    dataset = Image_Dataset(args)
    sampler_train = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed)
    data_loader = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=1,
        drop_last=False,
        collate_fn=lambda x: x,
        shuffle=False,
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'SAM {args.dataset} {args.split} split'
    for batch in metric_logger.log_every(data_loader, 10, header):
        sample = batch[0]
        img = sample["image"]
        if img is None:
            # skip images that can't be loaded
            continue
        else:
            masks = mask_generator.generate(img)
            h, w = img.shape[:2]
            total_mask = merge_anns(masks, h, w)
            image_array = np.uint8(contour(total_mask) * 255)
            img = Image.fromarray(image_array, 'L')
            img.save(sample["save_path"])

        metric_logger.synchronize_between_processes()